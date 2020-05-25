from __future__ import print_function

import protos.grpc_service_pb2 as grpc_service_pb2
import protos.grpc_service_pb2_grpc as grpc_service_pb2_grpc

import grpc
import numpy as np
import time

# Number of dummy images to send
N_DUMMY_IMAGES = 1000

INPUT_NODE_NAME = "data"
OUTPUT_NODE_NAME = "fc1000/Reshape_output"

STACK = True

IMAGE_LIST = "~/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/val.txt"
IMAGE_DIR = "~/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min"

import argparse
parser = argparse.ArgumentParser(description='Xilin ML Suit gRPC Client')
parser.add_argument('--batchsize', metavar="<batch size>", type=int,
                    help="Batch size", default=10)
parser.add_argument('-n', type=int, help="Number of images", default=490)
parser.add_argument("--address", metavar="<address>", type=str,
                    help="Server address", default="localhost")
parser.add_argument("-p", metavar="<port>", type=int,
                    help="Server port", default=5000)
args = parser.parse_args()

N_IMAGENET_IMAGES = args.n
BATCH_SIZE = args.batchsize
SERVER_ADDRESS = args.address
SERVER_PORT = args.p



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
    print("gRPC streaming enabled, batch size {batch}".format(batch=BATCH_SIZE))

    start_time = time.time()
    # Connect to server
    with grpc.insecure_channel('{address}:{port}'.format(address=SERVER_ADDRESS,
                                                         port=SERVER_PORT)) as channel:
        stub = grpc_service_pb2_grpc.GRPCServiceStub(channel)

        stub.Status(grpc_service_pb2.StatusRequest())

        # Make a call
        responses = stub.StreamInfer(empty_image_generator(n))

        for i, responses in enumerate(responses):
            if i % print_interval == 0:
                print(i)
    total_time = time.time() - start_time
    print("{n} images in {time} seconds ({speed} images/s)"
          .format(n=n,
                  time=total_time,
                  speed=float(n) / total_time))


if __name__ == '__main__':
    dummy_client(N_DUMMY_IMAGES)
