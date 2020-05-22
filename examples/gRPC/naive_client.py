from __future__ import print_function

import protos.grpc_service_pb2 as grpc_service_pb2
import protos.grpc_service_pb2_grpc as grpc_service_pb2_grpc

import request_wrapper

import grpc
import numpy as np
import time

# gRPC server info

SERVER_ADDRESS = "localhost"
SERVER_PORT = 5000

# Number of dummy images to send
N_DUMMY_IMAGES = 1000

INPUT_NODE_NAME = "data"
OUTPUT_NODE_NAME = "fc1000/Reshape_output"

BATCH_SIZE = 4

IMAGE_LIST = "~/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/val.txt"
IMAGE_DIR = "~/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min"



def empty_image_generator(n):
    '''
    Generate empty images

    n: number of images
    '''
    for _ in range(n // BATCH_SIZE):
        request = grpc_service_pb2.InferRequest()
        request.raw_input.append(np.zeros((BATCH_SIZE, 3, 224, 224), dtype=np.float32).tobytes())
        yield request


def dummy_client(n, print_interval=50):
    '''
    Start a dummy client

    n: number of images to send
    print_interval: print a number after this number of images is done
    '''
    print("Dummy client sending {n} images...".format(n=n))
    print("gRPC streaming disabled, batch size {batch}".format(batch=BATCH_SIZE))

    start_time = time.time()
    # Connect to server
    with grpc.insecure_channel('{address}:{port}'.format(address=SERVER_ADDRESS,
                                                         port=SERVER_PORT)) as channel:
        stub = grpc_service_pb2_grpc.GRPCServiceStub(channel)

        # Make a call
        for i in range(n // BATCH_SIZE):
            responses = stub.Infer(list(empty_image_generator(BATCH_SIZE))[0])
    total_time = time.time() - start_time
    print("{n} images in {time} seconds ({speed} images/s)"
          .format(n=n,
                  time=total_time,
                  speed=float(n) / total_time))


if __name__ == '__main__':
    dummy_client(N_DUMMY_IMAGES)
