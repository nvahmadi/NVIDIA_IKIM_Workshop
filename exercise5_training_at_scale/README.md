# Training@Scale

Code examples for the corresponding slide deck (version 4.0) by Maximilian Baust (NVIDIA). Please refer to `LICENSE.txt` for the evaluation license agreement.

## Running the Examples using the Pytorch Image from the NVIDIA GPU Cloud

`docker run --gpus all --ipc=host -it -p 9999:9999 -v /path/to/this/repo/:/workspace nvcr.io/nvidia/pytorch:22.03-py3`

## Running the Tiling & Tensor Core Utilization Example

`pip install torch_tb_profiler`

`python tiling_and_tensor_cores_pytorch.py` or alternatively

`python tiling_and_tensor_cores_pytorch.py --use_tensor_cores`

`tensorboard --logdir=./log --port=9999`

## Running the CUDA Graphs Example 

`python cuda_graphs.py` or alternatively

`python cuda_graphs.py --use_cuda_graphs`

## Running the Multi-GPU & Nsigth Example

`nsys profile --trace nvtx,cuda,cublas,cudnn python model_distribution_and_nvtx.py`

