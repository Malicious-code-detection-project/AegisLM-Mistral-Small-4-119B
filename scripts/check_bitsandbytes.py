import torch
from bitsandbytes.functional import dequantize_4bit, quantize_4bit

x = torch.randn(
    1024,
    1024,
    device="cuda",
    dtype=torch.bfloat16,
)

q, state = quantize_4bit(
    x,
    blocksize=64,
    compress_statistics=True,
    quant_type="nf4",
)

y = dequantize_4bit(q, quant_state=state)
torch.cuda.synchronize()

print("input:", x.shape, x.dtype)
print("quantized:", q.shape, q.dtype)
print("restored:", y.shape, y.dtype)
print("finite:", torch.isfinite(y).all().item())
print("BITSANDBYTES SMALL NF4 PASS")
