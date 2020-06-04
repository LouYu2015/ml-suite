from __future__ import print_function

import protos.grpc_service_pb2 as grpc_service_pb2
import protos.grpc_service_pb2_grpc as grpc_service_pb2_grpc


from sklearn import metrics
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
parser.add_argument("-p", metavar="<port>", type=int,
                    help="Server port", default=5000)
parser.add_argument("--stream", default=False, action="store_true")
args = parser.parse_args()

N_DUMMY_IMAGES = args.n
BATCH_SIZE = args.batchsize
SERVER_ADDRESS = args.address
SERVER_PORT = args.p
USE_STREAMING = args.stream


def imagenet_image_generator(file_name, n):
    import csv
    import os
    from PIL import Image
    reader = csv.reader(open(os.path.expanduser(file_name), "r"), delimiter=" ")
    for i, row in enumerate(reader):
        if i >= n:
            break
        image_path = row[0]
        file_name = os.path.expanduser(os.path.join(IMAGE_DIR, image_path))
        image = Image.open(file_name)
        image = image.resize((224, 224))
        image = np.asarray(image)
        if len(image.shape) == 2:
            image = np.stack([image]*3, axis=2)
        image = image - np.array([104.007, 116.669, 122.679])
        # image = image/255
        image = np.transpose(image, (2, 0, 1))
        yield image


def imagenet_label_generator(file_name, n):
    import csv
    import os
    reader = csv.reader(open(os.path.expanduser(file_name), "r"), delimiter=" ")
    for i, row in enumerate(reader):
        if i >= n:
            break
        label = row[1]
        yield int(label)


def imagenet_request_generator(file_name, n):
    try:
        i = 0
        data = []
        images = imagenet_image_generator(file_name, 490)
        images = itertools.cycle(images)
        images = itertools.islice(images, n)
        for image in images:
            data.append(image)
            i += 1

            if i == BATCH_SIZE:
                request = grpc_service_pb2.InferRequest()
                request.raw_input.append(np.array(data, dtype=np.float32).tobytes())
                yield request

                i = 0
                data = []
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e


def imagenet_client(file_name, n, print_interval=50):
    print("Sending {n} Imagenet images using batch size {batch_size}...".format(
        n=n,
        batch_size=BATCH_SIZE
    ))

    assert(n % BATCH_SIZE == 0)

    start_time = time.time()
    requests = list(imagenet_request_generator(file_name, n))
    total_time = time.time() - start_time
    print("Image load time: {time:.2f}".format(time=total_time))
    start_time = time.time()
    predictions = []
    # Connect to server
    with grpc.insecure_channel('{address}:{port}'.format(address=SERVER_ADDRESS,
                                                         port=SERVER_PORT)) as channel:
        stub = grpc_service_pb2_grpc.GRPCServiceStub(channel)

        # Make a call

        if USE_STREAMING:
            print("Using gRPC streaming")
            def it():
                for request in requests:
                    yield request
            responses = stub.StreamInfer(it())
        else:
            print("Not using gRPC streaming")
            responses = [stub.Infer(request) for request in requests]

        # Get responses
        for i, response in enumerate(responses):
            if i % print_interval == 0:
                print(i)
            response = np.frombuffer(response.raw_output[0], dtype=np.float32).reshape((-1, 1000))
            prediction = np.argmax(response, axis=1)
            predictions.append(prediction)
    total_time = time.time() - start_time
    print("Sent {n} images in {time:.3f} seconds ({speed:.3f} images/s), excluding image load time"
          .format(n=n,
                  time=total_time,
                  speed=float(n) / total_time))
    labels = imagenet_label_generator(file_name, 490)
    labels = itertools.cycle(labels)
    labels = itertools.islice(labels, n)
    labels = list(labels)
    # print(predictions)
    # print(labels)
    predictions = np.array(predictions).reshape((-1))
    labels = np.array(labels).reshape((-1))
    # print(predictions)
    # print(labels)
    print("Accuracy: {acc:.4}".format(acc=metrics.accuracy_score(labels, predictions)))


if __name__ == '__main__':
    imagenet_client(IMAGE_LIST, N_DUMMY_IMAGES)
