from __future__ import print_function

import protos.grpc_service_pb2 as grpc_service_pb2
import protos.grpc_service_pb2_grpc as grpc_service_pb2_grpc

import grpc
import numpy as np
import time
import itertools

# Number of dummy images to send
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
parser.add_argument("-p", metavar="<port>", type=int, nargs="+",
                    help="Server port", default=[5000])
args = parser.parse_args()

N_DUMMY_IMAGES = args.n
BATCH_SIZE = args.batchsize
SERVER_ADDRESS = args.address
ADDRESSES = ["{address}:{port}".format(address=SERVER_ADDRESS,
                                       port=port) for port in args.p]


def iterator_split(iterator, size):
    iterators = itertools.tee(iterator, size)
    result = []
    for i, it in enumerate(iterators):
        result.append(itertools.islice(it, i, None, size))
    return result


def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = itertools.cycle(iter(it).next for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = itertools.cycle(itertools.islice(nexts, pending))


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
    print("Connecting to", ADDRESSES)
    print("Batch size {batch}".format(batch=BATCH_SIZE))

    start_time = time.time()
    # Connect to server

    request_its = [empty_image_generator(n // len(ADDRESSES))
                   for _ in range(len(ADDRESSES))]
    channels = [grpc.insecure_channel(address)
                for address in ADDRESSES]
    try:
        results = [grpc_service_pb2_grpc.GRPCServiceStub(channel).StreamInfer(it)
                   for channel, it in zip(channels, request_its)]
        for i, response in enumerate(roundrobin(*results)):
            if i % print_interval == 0:
                print(i)
    finally:
        for channel in channels:
            channel.close()

    total_time = time.time() - start_time
    print("{n} images in {time:.1f} seconds ({speed:.1f} images/s)"
          .format(n=n,
                  time=total_time,
                  speed=float(n) / total_time))


if __name__ == '__main__':
    dummy_client(N_DUMMY_IMAGES)
