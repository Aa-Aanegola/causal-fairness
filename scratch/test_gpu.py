import torch
import time
import subprocess

rnd = torch.randn(1, 3, 224, 224).cuda()

# exec nvidia-smi to see which GPU is being used
subprocess.run("nvidia-smi", shell=True)