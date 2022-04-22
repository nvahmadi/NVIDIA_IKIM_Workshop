nsys profile \
     --delay 50 \
     --duration 30 \
     --output ./output_base \
     --force-overwrite true \
     --trace-fork-before-exec true \
     python3 train_evaluate_nvtx.py --optimized