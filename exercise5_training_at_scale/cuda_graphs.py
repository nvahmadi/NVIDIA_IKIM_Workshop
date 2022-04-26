import torch

import argparse
import os

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

K = 512

num_reps = 20
num_warmup = 3

class Layer(torch.nn.Module):
    def __init__(self):
        super(Layer, self).__init__()
        self.linear = torch.nn.Linear(K, K).to('cuda:0')
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x):
        y = self.linear(x)
        return x + self.dropout(y)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='training@scale - limiters and cuda graphs example')
    parser.add_argument('--use_cuda_graphs', default=False, action='store_true', help = 'use CUDA graphs to minimize kernel launch latency')
    
    args = parser.parse_args()

    # instanciate model, loss and optimizer
    model = torch.nn.Sequential(Layer(), Layer(), Layer(), Layer(), Layer()).cuda()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # placeholders used for capture
    static_input = torch.randn(K, K, device='cuda')
    static_target = torch.randn(K, K, device='cuda')

    # warmup (here, only with dummy data for convenience)
    cuda_stream = torch.cuda.Stream()
    cuda_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(cuda_stream):
        for i in range(num_warmup):
            optimizer.zero_grad(set_to_none=True)
            y_pred = model(static_input)
            loss = loss_fn(y_pred, static_target)
            loss.backward()
            optimizer.step()
    torch.cuda.current_stream().wait_stream(cuda_stream)

    # capture graph
    if args.use_cuda_graphs:
        cuda_graph = torch.cuda.CUDAGraph()
        # Sets grads to None before capture, so backward() will create
        # .grad attributes with allocations from the graph's private pool
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.graph(cuda_graph):
            static_pred = model(static_input)
            static_loss = loss_fn(static_pred, static_target)
            static_loss.backward()
            optimizer.step()

    real_inputs = [torch.rand_like(static_input) for _ in range(num_reps)]
    real_targets = [torch.rand_like(static_target) for _ in range(num_reps)]

    start.record()
    for data, target in zip(real_inputs, real_targets):
        
        if args.use_cuda_graphs:
                
            # filling input memory of graph
            static_input.copy_(data)
            static_target.copy_(target)
            
            # replay graph
            cuda_graph.replay()
        else:
            pred = model(data)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()

    end.record()
    torch.cuda.synchronize()

    elapsed_time = start.elapsed_time(end)
    if args.use_cuda_graphs:
        print('elapsed time using CUDA graphs: ' + str(elapsed_time) + ' milliseconds')
    else:
        print('elapsed time without CUDA graphs: ' + str(elapsed_time) + ' milliseconds')
