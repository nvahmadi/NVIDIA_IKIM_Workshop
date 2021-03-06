unzip /workspace/dataset/CAMELYON16.zip -d /workspace

nsys profile \
     --delay 50 \
     --duration 30 \
     --output ./output_base \
     --force-overwrite true \
     --trace-fork-before-exec true \
     python3 train_evaluate_nvtx.py --dataset "$WORKDIR/CAMELYON16/dataset_0_tumor_091.json" --root "$WORKDIR/CAMELYON16" --baseline
