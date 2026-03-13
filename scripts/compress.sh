export TRANSFORMERS_CACHE=/nfs/scistore19/alistgrp/huggingface/hub

python src/compress.py \
  --model Qwen/Qwen2-57B-A14B \
  --target-reduction 0.2 \
  --whitening output \
  --calib-dataset wikitext-2 \
  --calib-split train \
  --calib-max-samples 256