# A gRPC Inference Server

For SONIC support, please checkout branch [`sonic-grpc`](https://github.com/LouYu2015/ml-suite/tree/sonic-grpc/examples/gRPC).

To start the server in the docker image:

* Run `scripts/server_setup.sh` to quantize the model and install gRPC
* Run `./run.sh -t gRPC --batchsize 4 -m resnet50`

To start the client:

* Run `scripts/client_setup.sh`
* Change the server address in `client.py` and make sure that the batch
  size matches the server
* Run `client.py`