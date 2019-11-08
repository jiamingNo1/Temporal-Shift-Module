export CUDA_VISIBLE_DEVICES=0
python main.py \
     --mode test \
     --arch mobilenetv2 \
     --num_segments 8 \
     --batch_size 8 \
     --no_partialbn \
     --shift --shift_div=8