#!bin/bash

wget -O model.tgz https://www.dropbox.com/s/4bml3uzull0ckbu/model.tgz?dl=0
tar zxvf model.tgz
python3 inference.py --save_dir=./model/  --test_file=$1 --output=$2
