# A gRPC Inference Server

This is a gRPC inference server that provides a gRPC interface to ML Suite. The interface is compatible with the ``TensorRT`` package in [SONIC](https://github.com/hls-fpga-machine-learning/SonicCMS). Previous TensorRT SONIC clients can also connect to this server.

See the top-level README on how to download and start the ML Suite docker image.

To start the server in the docker image:

* Clone this repo
* Run `scripts/server_setup.sh` to quantize the model and install gRPC
    * Alternatively, follow the example in `ml-suite/examples/caffe` to manually generate the model.
* Run `./run.sh -t gRPC --batchsize <batch size> -m resnet50 --deviceid "<list of device ID>" --port <port>`.
  For example: `./run.sh -m resnet50 --batchsize 4 --deviceid "4 5 6 7" --port 5000`
     * `--batchsize`: number of images per request. Should be the same as the client's settings.
     * `-m`: model name
         * (To use models other than the example, edit `$NETCFG` and `$QUANTCFG` in `run.sh` to point to your own model)
     * `--deviceid`: a space-separated list of FPGA ID's that should be used.
     * `--port`: port to listen on

The naive client (dummy client) just sends empty images (totally black) to test and benchmark the server. To start the naive client:

* Run `scripts/client_setup.sh`
* Run `naive_client.py`. To see a list of options, run `naive_client.py -h`.

The SONIC client sends real jet data. Please refer to the SONIC documentation on how to use it.
