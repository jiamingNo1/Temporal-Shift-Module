python3 main.py \
     --mode train \
     --arch mobilenetv2 \
     --num_segments 8 \
     --update_weight 2 \
     --no_partialbn \
     --shift --shift_div=8
