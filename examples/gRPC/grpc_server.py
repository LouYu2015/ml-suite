from __future__ import print_function

import inference_server_pb2_grpc

import request_wrapper

STACK_CHANNELS = False
from xfdnn.rt import xdnn, xdnn_io
import numpy as np


class InferenceServicer(inference_server_pb2_grpc.InferenceServicer):
    '''
    This implements the inference service
    '''
    def __init__(self, fpgaRT, output_buffers, n_streams, input_shapes, fcWeight, fcBias, job_id_offsets):
        '''
        fpgaRT: fpga runtime
        output_buffers: a list of map from node name to numpy array.
           Store the output.
           The length should be equal to n_streams.
        n_streams: number of concurrent async calls
        input_shapes: map from node name to numpy array shape
        '''
        (self.fcWeight, self.fcBias) = (fcWeight, fcBias)
        self.fpgaRT = fpgaRT
        self.output_buffers = output_buffers
        self.n_streams = n_streams
        self.input_shapes = input_shapes
        self.job_id_offsets = job_id_offsets

    def push(self, request, in_slot):
        # Convert input format
        request = request_wrapper.protoToDict(request, self.input_shapes, stack=STACK_CHANNELS)

        # Send to FPGA
        self.fpgaRT.exec_async(request,
                               self.output_buffers[in_slot],
                               in_slot)

    def pop(self, out_slot):
        # Wait for finish signal
        self.fpgaRT.get_result(out_slot)

        # Read output
        response = self.output_buffers[out_slot]

        fcOutput = np.empty((response["fc1000/Reshape_output"].shape[0], 1000),
                                 dtype=np.float32, order='C')
        xdnn.computeFC(self.fcWeight, self.fcBias,
                       response["fc1000/Reshape_output"], fcOutput)
        response = request_wrapper.dictToProto({"fc1000/Reshape_output": fcOutput})
        return response

    def Inference(self, request_iterator, context):
        job_id_offset = self.job_id_offsets.get()
        try:
            in_slot = 0  # Next empty slot
            out_slot = 0  # Next ready slot
            for request in request_iterator:
                # Feed to FPGA
                self.push(request, job_id_offset + in_slot % self.n_streams)
                in_slot += 1

                # Start to pull output when the queue is full
                if in_slot - out_slot >= self.n_streams - 1:
                    yield self.pop(job_id_offset + out_slot % self.n_streams)
                    out_slot += 1

            # pull remaining output
            while in_slot - out_slot > 0:
                yield self.pop(job_id_offset + out_slot % self.n_streams)
                out_slot += 1
        finally:
            self.job_id_offsets.put(job_id_offset)
