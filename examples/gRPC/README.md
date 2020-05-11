# A gRPC Inference Server

To start the server in the docker image:

* Run `scripts/server_setup.sh` to quantize the model and install gRPC
* Run `./run.sh -t gRPC --batchsize 4 -m resnet50`

To start the client:

* Run `scripts/client_setup.sh`
* Change the server address in `client.py` and make sure that the batch
  size matches the server
* Run `client.py`