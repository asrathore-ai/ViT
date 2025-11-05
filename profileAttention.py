import torch 
from Attention import Transformer
import time 

# --- Profiling Setup ---
B = 4
seq_len = 768
d_model = 512
vocab_size = 37_000

# Use fewer runs for training, as backward pass is slow
WARMUP_RUNS = 5
TIMING_RUNS_INF = 100
TIMING_RUNS_TRAIN = 20

# --- Create Model ---
# We create one model and move it before each test
model = Transformer(vocab_size, seq_len)
print(f"Profiling Model (Batch={B}, Seq={seq_len}, Dim={d_model})\n")

# --- 1. Profile CPU Inference (Forward Pass) ---
print("--- 1. CPU Inference ---")
device = torch.device("cpu")
model.to(device)
a = torch.rand([B, seq_len, d_model], device=device)
b = torch.rand([B, seq_len, d_model], device=device)

with torch.no_grad(): # Disable gradient for inference
    for _ in range(WARMUP_RUNS):
        _ = model(a, b)
    
    start_time = time.perf_counter()
    for _ in range(TIMING_RUNS_INF):
        _ = model(a, b)
    end_time = time.perf_counter()

cpu_inf_time = (end_time - start_time) / TIMING_RUNS_INF * 1000 # in ms
print(f"Avg. CPU Inference: {cpu_inf_time:.2f} ms")


# --- 2. Profile MPS Inference (Forward Pass) ---
print("\n--- 2. MPS Inference ---")
device = torch.device("mps")
model.to(device)
a = a.to(device)
b = b.to(device)

with torch.no_grad():
    for _ in range(WARMUP_RUNS):
        _ = model(a, b)
    torch.mps.synchronize() # Wait for warm-up to finish
    
    start_time = time.perf_counter()
    for _ in range(TIMING_RUNS_INF):
        _ = model(a, b)
    torch.mps.synchronize() # MUST_HAVE: Wait for all runs to finish
    end_time = time.perf_counter()

mps_inf_time = (end_time - start_time) / TIMING_RUNS_INF * 1000 # in ms
print(f"Avg. MPS Inference: {mps_inf_time:.2f} ms")
print(f"-> MPS is {cpu_inf_time / mps_inf_time:.1f}x faster (Inference)")


# --- 3. Profile CPU Training (Forward + Backward) ---
print("\n--- 3. CPU Training ---")
device = torch.device("cpu")
model.to(device)
a = a.to(device)
b = b.to(device)

for _ in range(WARMUP_RUNS):
    o = model(a, b)
    loss = o.mean()
    loss.backward()
    model.zero_grad(set_to_none=True) # Clear grads

start_time = time.perf_counter()
for _ in range(TIMING_RUNS_TRAIN):
    o = model(a, b)
    loss = o.mean()
    loss.backward()
    model.zero_grad(set_to_none=True) # Clear grads
end_time = time.perf_counter()

cpu_train_time = (end_time - start_time) / TIMING_RUNS_TRAIN * 1000 # in ms
print(f"Avg. CPU Training: {cpu_train_time:.2f} ms")


# --- 4. Profile MPS Training (Forward + Backward) ---
print("\n--- 4. MPS Training ---")
device = torch.device("mps")
model.to(device)
a = a.to(device)
b = b.to(device)

for _ in range(WARMUP_RUNS):
    o = model(a, b)
    loss = o.mean()
    loss.backward()
    model.zero_grad(set_to_none=True)
torch.mps.synchronize() # Wait for warm-up to finish

start_time = time.perf_counter()
for _ in range(TIMING_RUNS_TRAIN):
    o = model(a, b)
    loss = o.mean()
    loss.backward()
    model.zero_grad(set_to_none=True)
torch.mps.synchronize() # MUST_HAVE: Wait for all runs to finish
end_time = time.perf_counter()

mps_train_time = (end_time - start_time) / TIMING_RUNS_TRAIN * 1000 # in ms
print(f"Avg. MPS Training: {mps_train_time:.2f} ms")
print(f"-> MPS is {cpu_train_time / mps_train_time:.1f}x faster (Training)")