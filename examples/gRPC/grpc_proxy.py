from __future__ import print_function

import protos.grpc_service_pb2_grpc as grpc_service_pb2_grpc
import protos.server_status_pb2 as server_status_pb2
import protos.model_config_pb2 as model_config_pb2
import protos.request_status_pb2 as request_status_pb2

import grpc
import multiprocessing as mp
from concurrent import futures
import itertools

import argparse
parser = argparse.ArgumentParser(description='Xilin ML Suit gRPC Inference Proxy')
parser.add_argument("--address", metavar="<address>", type=str,
                    help="Server address to forward to", default="localhost")
parser.add_argument("-p", "--ports", metavar="<port>", type=int, default=[5001], nargs="+",
                    help="Server ports to forward to")
parser.add_argument("-l", "--listen", metavar="<port>", type=int, default=5000,
                    help="Server ports to listen on")
args = parser.parse_args()

ADDRESSES = ["{address}:{port}".format(port=port, address=args.address)
             for port in args.ports]
PORT = args.listen


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


class InferenceServicer(grpc_service_pb2_grpc.GRPCServiceServicer):
    '''
    A proxy server that distributes requests
    '''
    def __init__(self, addresses):
        self.addresses = addresses
        self.next_address = 0

    def Status(self, request, context):
        master = self.addresses[0]
        with grpc.insecure_channel(master) as channel:
            stub = grpc_service_pb2_grpc.GRPCServiceStub(channel)
            return stub.Status(request)

    def Infer(self, request, context):
        master = self.addresses[self.next_address % len(self.addresses)]
        self.next_address += 1
        if self.next_address > 1000000:
            self.next_address = 0
        with grpc.insecure_channel(master) as channel:
            stub = grpc_service_pb2_grpc.GRPCServiceStub(channel)
            return stub.Infer(request)

    def StreamInfer(self, request_iterator, context):
        # request iterators
        request_its = iterator_split(request_iterator, len(self.addresses))
        channels = [grpc.insecure_channel(address)
                    for address in self.addresses]
        try:
            results = [grpc_service_pb2_grpc.GRPCServiceStub(channel).StreamInfer(it)
                       for channel, it in zip(channels, request_its)]
            for response in roundrobin(*results):
                yield response
        finally:
            for channel in channels:
                channel.close()

def main():
    print("Starting proxy on port", PORT)
    print("Addresses:", ADDRESSES)
    executor = futures.ThreadPoolExecutor(max_workers=64)
    executor.shutdown = lambda wait: None
    server = grpc.server(executor,
                         options=(('grpc.max_receive_message_length', 64*1024*1024),))
    servicer = InferenceServicer(ADDRESSES)
    grpc_service_pb2_grpc.add_GRPCServiceServicer_to_server(servicer,
                                                              server)

    # Bind port
    server.add_insecure_port('[::]:{port}'.format(port=PORT))

    # Start
    server.start()
    print("Server initialized")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        # Try to stop all threads
        print("Exiting due to keyboard interrupt")
        exit(0)


main()
