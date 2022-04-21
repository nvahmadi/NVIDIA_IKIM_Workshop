nsys profile \
     --delay 50 \
     --duration 30 \
     --output ./output_base \
     --force-overwrite true \
     --trace-fork-before-exec true \
     python3 train_evaluate_nvtx.py --dataset /data/Projects/Essen_Workshop/sources/data/CAMELYON16/dataset_0_tumor_091.json --root /data/Projects/Essen_Workshop/sources/data/CAMELYON16 --baseline