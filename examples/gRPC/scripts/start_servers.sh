BATCH_SIZE=4
./run.sh -m resnet50 --batchsize $BATCH_SIZE --deviceid "0" --port 5001&
./run.sh -m resnet50 --batchsize $BATCH_SIZE --deviceid "1" --port 5002&
./run.sh -m resnet50 --batchsize $BATCH_SIZE --deviceid "2" --port 5003&
./run.sh -m resnet50 --batchsize $BATCH_SIZE --deviceid "3" --port 5004&
./run.sh -m resnet50 --batchsize $BATCH_SIZE --deviceid "4" --port 5005&
./run.sh -m resnet50 --batchsize $BATCH_SIZE --deviceid "5" --port 5006&
./run.sh -m resnet50 --batchsize $BATCH_SIZE --deviceid "6" --port 5007&
./run.sh -m resnet50 --batchsize $BATCH_SIZE --deviceid "7" --port 5008&
