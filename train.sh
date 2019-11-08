python main.py \
     --mode train \
     --arch mobilenetv2 \
     --num_segments 8 \
     --batch_size 8 \
     --no_partialbn \
     --shift --shift_div=8
