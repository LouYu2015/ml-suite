from __future__ import print_function

import inference_server_pb2_grpc

import request_wrapper

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
            request, worker_id = request_queue.get()
            job_id = free_job_id_queue.get()

            # Convert input format
            request = request_wrapper.protoToDict(request, input_shapes, stack=STACK_CHANNELS)

            # Send to FPGA
            fpgaRT.exec_async(request,
                              output_buffers[job_id],
                              job_id)

            # Send to waiter
            occupied_job_id_queue.put((job_id, worker_id))
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
            job_id, worker_id = occupied_job_id_queue.get()

            # Wait for FPGA to finish
            fpgaRT.get_result(job_id)

            # Read output
            response = output_buffers[job_id]

            # Compute fully connected layer
            fcOutput = np.empty((response[output_node_name].shape[0], 1000),
                                     dtype=np.float32, order='C')
            xdnn.computeFC(fcWeight, fcBias,
                           response[output_node_name], fcOutput)

            # Send response
            response = request_wrapper.dictToProto({output_node_name: fcOutput})
            response_queues[worker_id].put(response)

            # Free job ID
            free_job_id_queue.put(job_id)
    except Exception as e:
        import traceback
        import sys
        traceback.print_exc()
        sys.exit()


class InferenceServicer(inference_server_pb2_grpc.InferenceServicer):
    '''
    This implements the inference service
    '''
    def __init__(self, fpgaRT, output_buffers, output_node_name, n_streams, input_shapes,
                 fcWeight, fcBias, n_workers):
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

    def Inference(self, request_iterator, context):
        # Assign worker ID
        worker_id = self.worker_id_queue.get()
        try:
            n_response_waiting = 0  # Number of pending responses
            for request in request_iterator:
                # Feed to FPGA
                self.request_queue.put((request, worker_id))
                n_response_waiting += 1

                # Send response when ready
                try:
                    while True:
                        response = self.response_queues[worker_id].get_nowait()
                        yield response
                        n_response_waiting -= 1
                except Queue.Empty:
                    pass

            # pull remaining output
            while n_response_waiting > 0:
                response = self.response_queues[worker_id].get()
                yield response
                n_response_waiting -= 1
        finally:
            # Return worker ID
            self.worker_id_queue.put(worker_id)
