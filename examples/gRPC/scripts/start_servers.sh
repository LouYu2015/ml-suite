BATCH_SIZE=4
./run.sh -m resnet50 --batchsize $BATCH_SIZE --deviceid "0 1" --port 5001&
./run.sh -m resnet50 --batchsize $BATCH_SIZE --deviceid "2 3" --port 5002&
./run.sh -m resnet50 --batchsize $BATCH_SIZE --deviceid "4 5" --port 5003&
./run.sh -m resnet50 --batchsize $BATCH_SIZE --deviceid "6 7" --port 5004&
