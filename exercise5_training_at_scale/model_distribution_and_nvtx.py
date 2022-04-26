import torch
from torch.nn import Linear, ReLU, MSELoss
from torch.cuda.nvtx import range_push, range_pop

input_dim = 1024
hidden_dim = 2048
output_dim = 1

batch_size = 512
split_size = 128

num_epochs = 10

class Model(torch.nn.Module):
    def __init__(self):

        super(Model, self).__init__()

        self.part1 = Linear(input_dim, hidden_dim).to('cuda:0')
        self.part2 = Linear(hidden_dim, output_dim).to('cuda:1')

    def forward(self, x):

        data_shards = iter(x.split(split_size, dim=0))

        next_shard = next(data_shards)
        prev_shard = self.part1(next_shard).to('cuda:1')
        output_list = []

        for next_shard in data_shards:

            output_list.append(self.part2(prev_shard))
            prev_shard = self.part1(next_shard).to('cuda:1')

        output_list.append(self.part2(prev_shard))

        return torch.cat(output_list)

if __name__ == "__main__":

    model = Model()
    loss_fn = MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    range_push("data_creation")
    data = torch.randn(batch_size, input_dim, device='cpu')
    labels = torch.randn(batch_size, output_dim, device='cpu')
    range_pop()

    range_push("host_2_device")
    data = data.to('cuda:0')
    labels = labels.to('cuda:1')
    range_pop()

    for e in range(num_epochs):

        range_push("zero_gradient_buffers")
        optimizer.zero_grad()
        range_pop()

        range_push("forward_pass")
        outputs = model(data)
        range_pop()

        range_push("loss_computation")
        loss = loss_fn(outputs, labels)
        range_pop()

        range_push("back_propagation")
        loss.backward()
        range_pop()

        range_push("model_update")
        optimizer.step()
        range_pop()
