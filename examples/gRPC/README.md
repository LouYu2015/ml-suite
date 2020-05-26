# A gRPC Inference Server

To start the server in the docker image:

* Run `scripts/server_setup.sh` to quantize the model and install gRPC
* Run `./run.sh -t gRPC --batchsize <batch size> -m resnet50 --deviceid "<list of device ID>" --port <port>`.
  For example: `./run.sh -m resnet50 --batchsize 4 --deviceid "4 5 6 7" --port 5000`

To start the client:

* Run `scripts/client_setup.sh`
* Change the server address in `client.py` and make sure that the batch
  size matches the server
* Run `naive_client.py`. To see a list of options, run `naive_client.py -h`.