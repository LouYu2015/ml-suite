sudo python -m pip install grpcio-tools
sudo python -m pip install Pillow
sudo python -m pip install ck

python -m ck pull repo:ck-env
python -m ck install package:imagenet-2012-val-min
python -m ck install package:imagenet-2012-aux