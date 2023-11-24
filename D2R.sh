#OFFICE-31
python D2R.py --gpu_id 0 --seed 2021 --output_src ./ckps/src --dset office --s 0 --da uda --net_src resnet50 --max_epoch 50
python D2R.py --gpu_id 0 --seed 2021 --output_src ./ckps/src --dset office --s 1 --da uda --net_src resnet50 --max_epoch 50
python D2R.py --gpu_id 0 --seed 2021 --output_src ./ckps/src --dset office --s 2 --da uda --net_src resnet50 --max_epoch 50
python D2R.py --gpu_id 0 --seed 2021 --output_src ./ckps/src --dset office --s 0 --da uda --net_src resnet50 --max_epoch 50 --dist_epoch 30 -net resnet50 --output ./ckps/tar --distill --topk 1 --batch_size 32
python D2R.py --gpu_id 0 --seed 2021 --output_src ./ckps/src --dset office --s 1 --da uda --net_src resnet50 --max_epoch 50 --dist_epoch 30 --net resnet50 --output ./ckps/tar --distill --topk 1 --batch_size 32
python D2R.py --gpu_id 0 --seed 2021 --output_src ./ckps/src --dset office --s 2 --da uda --net_src resnet50 --max_epoch 50 --dist_epoch 30 --net resnet50 --output ./ckps/tar --distill --topk 1 --batch_size 32

#OFFICE-HOME
python D2R.py --gpu_id 0 --seed 2021 --output_src ./ckps/src --dset office-home --s 0 --da uda --net_src resnet50 --max_epoch 50
python D2R.py --gpu_id 0 --seed 2021 --output_src ./ckps/src --dset office-home --s 1 --da uda --net_src resnet50 --max_epoch 50
python D2R.py --gpu_id 0 --seed 2021 --output_src ./ckps/src --dset office-home --s 2 --da uda --net_src resnet50 --max_epoch 50
python D2R.py --gpu_id 0 --seed 2021 --output_src ./ckps/src --dset office-home --s 3 --da uda --net_src resnet50 --max_epoch 50
python D2R.py --gpu_id 0 --seed 2021 --output_src ./ckps/src --dset office-home --s 0 --da uda --net_src resnet50 --max_epoch 50 --dist_epoch 30 --net resnet50 --output ./ckps/tar --distill --topk 1 --batch_size 32
python D2R.py --gpu_id 0 --seed 2021 --output_src ./ckps/src --dset office-home --s 1 --da uda --net_src resnet50 --max_epoch 50 --dist_epoch 30 --net resnet50 --output ./ckps/tar --distill --topk 1 --batch_size 32
python D2R.py --gpu_id 0 --seed 2021 --output_src ./ckps/src --dset office-home --s 2 --da uda --net_src resnet50 --max_epoch 50 --dist_epoch 30 --net resnet50 --output ./ckps/tar --distill --topk 1 --batch_size 32
python D2R.py --gpu_id 0 --seed 2021 --output_src ./ckps/src --dset office-home --s 3 --da uda --net_src resnet50 --max_epoch 50 --dist_epoch 30 --net resnet50 --output ./ckps/tar --distill --topk 1 --batch_size 32

#VISDA-C
python D2R.py --gpu_id 0 --seed 2021 --output_src ./ckps/src --dset VISDA-C --s 0 --da uda --net_src resnet101 --max_epoch 10 --batch_size 64 --lr 1e-3
python D2R.py --gpu_id 0 --seed 2021 --output_src ./ckps/src --dset VISDA-C --s 0 --da uda --net_src resnet101 --max_epoch 30 --dist_epoch 10 --net resnet101 --output ./ckps/tar --distill --topk 1 --batch_size 32  --lr 1e-3
