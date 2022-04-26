import torch
import torch.profiler

import argparse
import os

if not os.path.exists('./log'):
    os.makedirs('./log')

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

K = 4096
M = 27648

num_reps = 20

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='training@scale - tiling and tensor core example')
    parser.add_argument('--use_tensor_cores', default=False, action='store_true', help = 'run training in FP16 (to use tensor cores)')
    
    args = parser.parse_args()

    prec = torch.torch.float32
    if args.use_tensor_cores:
        print('Executing in FP16:')
        prec = torch.torch.float16
        logdir = './log/fp16'
        if not os.path.exists(logdir):
            os.makedirs(logdir)
    else:
        print('Executing in FP32:')
        logdir = './log/fp32'
        if not os.path.exists(logdir):
            os.makedirs(logdir)

    prof_schedule = torch.profiler.schedule(wait=0, warmup=0, active=34, repeat=0)
    callback = torch.profiler.tensorboard_trace_handler(logdir)
    prof = torch.profiler.profile(schedule=prof_schedule,
                                  on_trace_ready=callback,
                                  record_shapes=True,
                                  with_stack=True)

    prof.start()
    for step in range(34):

        N = 128 + step * 16

        A = torch.randn(M, K, dtype=prec, device='cuda')
        B = torch.randn(K, N, dtype=prec, device='cuda')

        runtime = 0.
        for r in range(num_reps):
            start.record()
            C = torch.matmul(A, B)
            end.record()
            torch.cuda.synchronize()
            elapsed_time = start.elapsed_time(end)
            if r > 0:
                runtime += start.elapsed_time(end)
        prof.step()
        
        runtime /= (num_reps - 1)

        if step % 8 == 1:
            print('\n---------------------------')

        print('K=' + str(N) + ': ' + str(runtime) + ' milliseconds')
        
    prof.stop()

