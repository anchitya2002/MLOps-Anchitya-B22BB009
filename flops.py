import torch
from fvcore.nn import FlopCountAnalysis
from model import SimpleCNN

model = SimpleCNN()
dummy_input = torch.randn(1, 3, 32, 32)

flops = FlopCountAnalysis(model, dummy_input)
print("Total FLOPs:", flops.total())
