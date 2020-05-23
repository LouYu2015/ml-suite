from __future__ import print_function

import inference_server_pb2_grpc

import request_wrapper

import grpc
import numpy as np
from sklearn import metrics
import time

# gRPC server info

SERVER_ADDRESS = "localhost"
SERVER_PORT = 5000

# Number of dummy images to send
N_DUMMY_IMAGES = 1000
N_IMAGENET_IMAGES = 490

INPUT_NODE_NAME = "data"
OUTPUT_NODE_NAME = "fc1000/Reshape_output"

STACK = True
BATCH_SIZE = 10

IMAGE_LIST = "~/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/val.txt"
IMAGE_DIR = "~/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min"



def empty_image_generator(n):
    '''
    Generate empty images

    n: number of images
    '''
    for _ in range(n // BATCH_SIZE):
        if STACK:
            request = {INPUT_NODE_NAME: np.zeros((BATCH_SIZE, 224, 224), dtype=np.float32)}
        else:
            request = {INPUT_NODE_NAME: np.zeros((BATCH_SIZE, 3, 224, 224), dtype=np.float32)}
        request = request_wrapper.dictToProto(request)
        yield request


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
        for image in imagenet_image_generator(file_name, n):
            data.append(image)
            i += 1

            if i == BATCH_SIZE:
                request = {INPUT_NODE_NAME: np.array(data, dtype=np.float32)}
                yield request_wrapper.dictToProto(request)

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
                                                         port=SERVER_PORT),
                               options=[
                                   ('grpc.max_send_message_length', 50 * 1024 * 1024),
                                   ('grpc.max_receive_message_length', 50 * 1024 * 1024)]
                               ) as channel:
        stub = inference_server_pb2_grpc.InferenceStub(channel)

        # Make a call
        def it():
            for request in requests:
                yield request
        responses = stub.Inference(it())

        # Get responses
        for i, response in enumerate(responses):
            if i % print_interval == 0:
                print(i)
            response = request_wrapper.protoToDict(response,
                                              {OUTPUT_NODE_NAME: (BATCH_SIZE, 1000)})
            prediction = np.argmax(response[OUTPUT_NODE_NAME], axis=1)
            predictions.append(prediction)
    total_time = time.time() - start_time
    print("Sent {n} images in {time:.3f} seconds ({speed:.3f} images/s), excluding image load time"
          .format(n=n,
                  time=total_time,
                  speed=float(n) / total_time))
    labels = list(imagenet_label_generator(file_name, n))
    # print(predictions)
    # print(labels)
    predictions = np.array(predictions).reshape((-1))
    labels = np.array(labels).reshape((-1))
    # print(predictions)
    # print(labels)
    print("Accuracy: {acc:.4}".format(acc=metrics.accuracy_score(labels, predictions)))


def dummy_client(n, print_interval=50):
    '''
    Start a dummy client

    n: number of images to send
    print_interval: print a number after this number of images is done
    '''
    print("Dummy client sending {n} images...".format(n=n))

    start_time = time.time()
    # Connect to server
    with grpc.insecure_channel('{address}:{port}'.format(address=SERVER_ADDRESS,
                                                         port=SERVER_PORT)) as channel:
        stub = inference_server_pb2_grpc.InferenceStub(channel)

        # Make a call
        responses = stub.Inference(empty_image_generator(n))

        # Get responses
        for i, response in enumerate(responses):
            if i % print_interval == 0:
                print(i)
            # print(request_wrapper.protoToDict(response,
            #                                   {"loss3_classifier/Reshape_output": (1024)}))
    total_time = time.time() - start_time
    print("{n} images in {time} seconds ({speed} images/s)"
          .format(n=n,
                  time=total_time,
                  speed=float(n) / total_time))


if __name__ == '__main__':
    # dummy_client(N_DUMMY_IMAGES)
    imagenet_client(IMAGE_LIST, N_IMAGENET_IMAGES)
