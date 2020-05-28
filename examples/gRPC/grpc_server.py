from __future__ import print_function

import protos.grpc_service_pb2_grpc
import protos.grpc_service_pb2 as grpc_service_pb2
import protos.server_status_pb2 as server_status_pb2
import protos.request_status_pb2 as request_status_pb2
import protos.model_config_pb2 as model_config_pb2

STACK_CHANNELS = False
from xfdnn.rt import xdnn, xdnn_io
import numpy as np
import multiprocessing as mp
import threading
import Queue


def fpga_worker(fpgaRT, output_buffers, input_shapes,
                free_job_id_queue, occupied_job_id_queue, request_queue):
    """
    Puts request into FPGA
    """
    try:
        while True:
            request, worker_id, request_id = request_queue.get()
            job_id = free_job_id_queue.get()

            # Convert input format
            array = np.frombuffer(request, dtype=np.float32)
            try:
                array = array.reshape(list(input_shapes.values())[0])
            except ValueError:
                free_job_id_queue.put(job_id)
                continue

            input_buffer = {list(input_shapes.keys())[0]: array}

            # Send to FPGA
            fpgaRT.exec_async(input_buffer,
                              output_buffers[job_id],
                              job_id)

            # Send to waiter
            occupied_job_id_queue.put((job_id, worker_id, request_id))
    except Exception as e:
        import traceback
        import sys
        traceback.print_exc()
        sys.exit()


def fpga_waiter(fpgaRT, output_buffers, output_node_name, fcWeight, fcBias,
                free_job_id_queue, occupied_job_id_queue, response_queues):
    """
    Wait for job to finish and distribute result to workers
    """
    try:
        while True:
            job_id, worker_id, request_id = occupied_job_id_queue.get()

            # Wait for FPGA to finish
            fpgaRT.get_result(job_id)

            # Read output
            response = output_buffers[job_id]

            # Compute fully connected layer
            fcOutput = np.empty((response[output_node_name].shape[0], 1000),
                                     dtype=np.float32, order='C')
            xdnn.computeFC(fcWeight, fcBias,
                           response[output_node_name], fcOutput)

            # Give response to worker
            response_queues[worker_id].put((fcOutput.tobytes(), request_id))

            # Free job ID
            free_job_id_queue.put(job_id)
    except Exception as e:
        import traceback
        import sys
        traceback.print_exc()
        sys.exit()


class InferenceServicer(protos.grpc_service_pb2_grpc.GRPCServiceServicer):
    '''
    This implements the inference service
    '''
    def __init__(self, fpgaRT, output_buffers, output_node_name, n_streams, input_shapes,
                 fcWeight, fcBias, n_workers, batch_size):
        '''
        fpgaRT: fpga runtime
        output_buffers: a list of map from node name to numpy array.
           Store the output.
           The length should be equal to n_streams.
        n_streams: number of concurrent async calls
        input_shapes: map from node name to numpy array shape
        fcWeight: final fully connected layer weights
        fcBias: final fully connected layer bias
        '''
        (self.fcWeight, self.fcBias) = (fcWeight, fcBias)
        self.fpgaRT = fpgaRT
        self.output_buffers = output_buffers
        self.n_streams = n_streams
        self.input_shapes = input_shapes
        self.batch_size = batch_size

        # Queue of free job ID
        free_job_id_queue = mp.Queue()
        for job_id in range(self.n_streams):
            free_job_id_queue.put(job_id)

        # Queue of occupied job ID
        occupied_job_id_queue = mp.Queue()

        # Queue of request
        request_queue = mp.Queue(n_streams)
        self.request_queue = request_queue

        # Queue of response for each worker
        response_queues = [mp.Queue() for _ in range(n_workers)]
        self.response_queues = response_queues

        # Queue of free worker ID
        self.worker_id_queue = mp.Queue()
        for worker_id in range(n_workers):
            self.worker_id_queue.put(worker_id)

        # Start worker
        t = threading.Thread(target=fpga_worker,
                             args=(fpgaRT, output_buffers, input_shapes,
                                   free_job_id_queue, occupied_job_id_queue, request_queue))
        t.daemon = True
        t.start()

        # Start waiter
        t = threading.Thread(target=fpga_waiter,
                             args=(fpgaRT, output_buffers, output_node_name, fcWeight, fcBias,
                                   free_job_id_queue, occupied_job_id_queue, response_queues))
        t.daemon = True
        t.start()

    def Status(self, request, context):
        # Model status
        server_status = server_status_pb2.ServerStatus()
        server_status.id = "inference:0"
        config = server_status.model_status["resnet50_netdef"].config
        config.max_batch_size = 16
        config.name = "resnet50_netdef"

        # Input
        input = config.input.add()
        input.name = "input"
        input.data_type = model_config_pb2.TYPE_FP32
        input.dims.append(3*224*224)

        # Output
        output = config.output.add()
        output.name = "output/BiasAdd"
        output.data_type = model_config_pb2.TYPE_FP32
        output.dims.append(self.batch_size)
        output.dims.append(1000)

        request_status = request_status_pb2.RequestStatus(
            code=request_status_pb2.SUCCESS)

        return grpc_service_pb2.StatusResponse(server_status=server_status,
                                               request_status=request_status)

    def Infer(self, request, context):
        responses = list(self.StreamInfer([request], context))
        return responses[0]

    def pull_response(self, worker_id, wait=True):
        """
        Pull a response for worker
        """
        if wait:
            fcOutput, request_id = self.response_queues[worker_id].get()
        else:
            fcOutput, request_id = self.response_queues[worker_id].get_nowait()

        # Construct response
        request_status = request_status_pb2.RequestStatus(
            code=request_status_pb2.SUCCESS,
            server_id="inference:0")
        reply = grpc_service_pb2.InferResponse(request_status=request_status)
        reply.meta_data.id = request_id
        reply.meta_data.model_version = -1
        reply.meta_data.batch_size = self.batch_size

        # Construct output
        output = reply.meta_data.output.add()
        output.name = "output/BiasAdd"
        output.raw.dims.append(self.batch_size)
        output.raw.dims.append(1000)
        output.raw.batch_byte_size = 1000*self.batch_size*4
        reply.raw_output.append(fcOutput)

        return reply

    def StreamInfer(self, request_iterator, context):
        # Assign worker ID
        worker_id = self.worker_id_queue.get()
        try:
            n_response_waiting = 0  # Number of pending responses
            for request in request_iterator:
                # Feed to FPGA
                self.request_queue.put((request.raw_input[0], worker_id, request.meta_data.id))
                n_response_waiting += 1

                # Send response when ready
                try:
                    while True:
                        yield self.pull_response(worker_id, wait=False)
                        n_response_waiting -= 1
                except Queue.Empty:
                    pass

            # pull remaining output
            while n_response_waiting > 0:
                yield self.pull_response(worker_id)
                n_response_waiting -= 1
        finally:
            # Return worker ID
            self.worker_id_queue.put(worker_id)
