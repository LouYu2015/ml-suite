from __future__ import print_function

import inference_server_pb2_grpc

import request_wrapper

STACK_CHANNELS = False
from xfdnn.rt import xdnn, xdnn_io
import numpy as np
import multiprocessing as mp
import Queue


def fpga_worker(fpgaRT, output_buffers, input_shapes,
                free_job_id_queue, occupied_job_id_queue, request_queue):
    """
    Puts request into FPGA
    """
    while True:
        request, worker_id = request_queue.get()
        print("Request from", worker_id)
        job_id = free_job_id_queue.get()
        print("Assign job ID", job_id)

        # Convert input format
        request = request_wrapper.protoToDict(request, input_shapes, stack=STACK_CHANNELS)

        # Send to FPGA
        print("Execute job", job_id)
        fpgaRT.exec_async(request,
                          output_buffers[job_id],
                          job_id)

        # Send to waiter
        print("Sent job", job_id,"to waiter")
        occupied_job_id_queue.put((job_id, worker_id))

def fpga_waiter(fpgaRT, output_buffers, fcWeight, fcBias,
                free_job_id_queue, occupied_job_id_queue, response_queues):
    """
    Wait for job to finish and distribute result to workers
    """
    while True:
        job_id, worker_id = occupied_job_id_queue.get()
        print("Wait for {job} from worker {worker}".format(job=job_id, worker=worker_id))

        # Wait for FPGA to finish
        fpgaRT.get_result(job_id)

        # Read output
        response = output_buffers[job_id]

        # Compute fully connected layer
        fcOutput = np.empty((response["fc1000/Reshape_output"].shape[0], 1000),
                                 dtype=np.float32, order='C')
        xdnn.computeFC(fcWeight, fcBias,
                       response["fc1000/Reshape_output"], fcOutput)

        # Send response
        response = request_wrapper.dictToProto({"fc1000/Reshape_output": fcOutput})
        print("Give response to", worker_id)
        response_queues[worker_id].put(response)

        # Free job ID
        free_job_id_queue.put(job_id)


class InferenceServicer(inference_server_pb2_grpc.InferenceServicer):
    '''
    This implements the inference service
    '''
    def __init__(self, fpgaRT, output_buffers, n_streams, input_shapes,
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
        mp.Process(target=fpga_worker,
                   args=(fpgaRT, output_buffers, input_shapes,
                         free_job_id_queue, occupied_job_id_queue, request_queue)) \
            .start()

        # Start waiter
        mp.Process(target=fpga_waiter,
                   args=(fpgaRT, output_buffers, fcWeight, fcBias,
                         free_job_id_queue, occupied_job_id_queue, response_queues)) \
            .start()

    def Inference(self, request_iterator, context):
        # Assign worker ID
        worker_id = self.worker_id_queue.get()
        print("Worker", worker_id, "started")
        try:
            n_response_waiting = 0  # Number of pending responses
            for request in request_iterator:
                # Feed to FPGA
                print("Put request")
                self.request_queue.put((request, worker_id))
                n_response_waiting += 1

                # Send response when ready
                try:
                    while True:
                        response = self.response_queues[worker_id].get_nowait()
                        yield response
                        print("Sent response")
                        n_response_waiting -= 1
                except Queue.Empty:
                    print("Queue empty")
                    pass

            # pull remaining output
            while n_response_waiting > 0:
                response = self.response_queues[worker_id].get()
                yield response
                n_response_waiting -= 1
        finally:
            # Return worker ID
            self.worker_id_queue.put(worker_id)
