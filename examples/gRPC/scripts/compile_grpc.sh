cd protodef/

# Generates inference_server_pb2.py and inference_server_pb2_grpc.py
python3 -m grpc_tools.protoc \
  -I./ \
  --python_out=../protos/ \
  --grpc_python_out=../protos/ \
  *.proto
