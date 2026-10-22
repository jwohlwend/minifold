import torch
import torch.nn.functional as F
from minifold.model.kernels._triton import triton, tl
import gc
import time


@triton.autotune(
    configs=[triton.Config({}, num_stages=0)],
    key=['C1', 'C2'],
)
@triton.jit
def inference_kernel(
    X_ptr,
    W1_ptr,
    W2_ptr,
    b1_ptr,
    b2_ptr,
    wn_ptr,
    bn_ptr,
    O_ptr,
    M: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_C2: tl.constexpr,
):
    # Get program ids
    pid_m = tl.program_id(0)

    # Create block pointers
    x_block_ptr = tl.make_block_ptr(
        base=X_ptr,
        shape=(M, C1),
        strides=(C1, 1),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, C1),
        order=(1, 0),
    )

    w1_block_ptr = tl.make_block_ptr(
        base=W1_ptr,
        shape=(C1, C2),
        strides=(C2, 1),
        offsets=(0, 0),
        block_shape=(C1, BLOCK_SIZE_C2),
        order=(1, 0),
    )

    b1_block_ptr = tl.make_block_ptr(
        base=b1_ptr,
        shape=(C2,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_C2,),
        order=(0,),
    )

    w2_block_ptr = tl.make_block_ptr(
        base=W2_ptr,
        shape=(C1, C2),
        strides=(C2, 1),
        offsets=(0, 0),
        block_shape=(C1, BLOCK_SIZE_C2),
        order=(1, 0),
    )

    b2_block_ptr = tl.make_block_ptr(
        base=b2_ptr,
        shape=(C2,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_C2,),
        order=(0,),
    )

    wn_block_ptr = tl.make_block_ptr(
        base=wn_ptr,
        shape=(C1,),
        strides=(1,),
        offsets=(0,),
        block_shape=(C1,),
        order=(0,),
    )

    bn_block_ptr = tl.make_block_ptr(
        base=bn_ptr,
        shape=(C1,),
        strides=(1,),
        offsets=(0,),
        block_shape=(C1,),
        order=(0,),
    )

    o_block_ptr = tl.make_block_ptr(
        base=O_ptr,
        shape=(M, C2),
        strides=(C2, 1),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_C2),
        order=(1, 0),
    )

    # Load data
    x = tl.load(x_block_ptr)
    dtype = x.dtype

    # Normalize
    x = x.to(tl.float32)
    x_mean = tl.sum(x, axis=1) / C1
    x -= x_mean[:, None]

    x_var = tl.sum((x * x), axis=1) / C1
    x_var = 1 / tl.sqrt(x_var + 1e-5)
    x *= x_var[:, None]

    # Scale and shift
    x = x.to(dtype)
    x *= tl.load(wn_block_ptr).to(dtype)
    x += tl.load(bn_block_ptr).to(dtype)

    # Compute gating
    for _ in range(0, tl.cdiv(C2, BLOCK_SIZE_C2)):
        # Compute gate
        w = tl.load(w1_block_ptr).to(dtype)
        b = tl.load(b1_block_ptr).to(dtype)
        g = tl.dot(x, w, allow_tf32=False)
        g += b[None, :]
        g = tl.sigmoid(g)
        g = g.to(dtype)

        # Compute output
        w = tl.load(w2_block_ptr).to(dtype)
        b = tl.load(b2_block_ptr).to(dtype)
        o = tl.dot(x, w, allow_tf32=False)
        o = o.to(dtype)
        o += b[None, :]
        o *= g

        # Store output
        tl.store(o_block_ptr, o)

        # Advance pointers
        w1_block_ptr = tl.advance(w1_block_ptr, (0, BLOCK_SIZE_C2))
        w2_block_ptr = tl.advance(w2_block_ptr, (0, BLOCK_SIZE_C2))
        b1_block_ptr = tl.advance(b1_block_ptr, (BLOCK_SIZE_C2,))
        b2_block_ptr = tl.advance(b2_block_ptr, (BLOCK_SIZE_C2,))
        o_block_ptr = tl.advance(o_block_ptr, (0, BLOCK_SIZE_C2))


def gating_kernel(X, W1, W2, b1, b2, wn, bn):
    if X.device.type == "cuda":
        B, N1, N2, C1 = X.shape
        M = B * N1 * N2
        C1, C2 = W1.shape

        X = X.contiguous()
        O = torch.empty((B, N1, N2, C2), device=X.device, dtype=X.dtype)

        BLOCK_SIZE_M = 64
        BLOCK_SIZE_C2 = 64
        assert M >= BLOCK_SIZE_M
        assert M % BLOCK_SIZE_M == 0
        assert C2 >= BLOCK_SIZE_C2
        assert C2 % BLOCK_SIZE_C2 == 0
        assert C1 <= 128

        grid = lambda META: (triton.cdiv(M, BLOCK_SIZE_M),)
        inference_kernel[grid](
            X,
            W1,
            W2,
            b1,
            b2,
            wn,
            bn,
            O,
            M,
            C1,
            C2,
            BLOCK_SIZE_M,
            BLOCK_SIZE_C2,
        )
        return O
    else:
        # Use unfused fallback for MPS/CPU
        x = F.layer_norm(X, X.shape[-1:], weight=wn, bias=bn, eps=1e-5)
        o = F.linear(x, W1, b1)
        o.sigmoid_()
        o *= F.linear(x, W2, b2)
        return o


#---------------- TESTS ----------------#

def fused(x, W1, W2, b1, b2, wn, bn):
    result = gating_kernel(x, W1, W2, b1, b2, wn, bn)
    return result


def unfused(x, W1, W2, b1, b2, wn, bn):
    x = F.layer_norm(x, x.shape[-1:], weight=wn, bias=bn, eps=1e-5)
    o = F.linear(x, W1, b1)
    o.sigmoid_()
    o *= F.linear(x, W2, b2)
    return o


# @torch.compile(dynamic=False, fullgraph=True)
def compiled(x, W1, W2, b1, b2, wn, bn):
    x = F.layer_norm(x, x.shape[-1:], weight=wn, bias=bn, eps=1e-5)
    o = F.linear(x, W1, b1)
    o.sigmoid_()
    o *= F.linear(x, W2, b2)
    return o


def create_input(device, dtype=torch.float32, grad=False, size=256):
    B = 1
    C = 128
    H = 128
    N = size

    x = 0.1 * torch.randn((B, N, N, C), device=device, dtype=dtype)
    W1 = 0.1 * torch.randn((C, H), device=device, dtype=dtype)
    W2 = 0.1 * torch.randn((C, H), device=device, dtype=dtype)
    b1 = 0.1 * torch.randn((H,), device=device, dtype=dtype)
    b2 = 0.1 * torch.randn((H,), device=device, dtype=dtype)
    wn = 0.1 * torch.randn((C,), device=device, dtype=dtype)
    bn = 0.1 * torch.randn((C,), device=device, dtype=dtype)

    x.requires_grad = grad
    W1.requires_grad = grad
    W2.requires_grad = grad
    b1.requires_grad = grad
    b2.requires_grad = grad
    wn.requires_grad = grad
    bn.requires_grad = grad

    return x, W1, W2, b1, b2, wn, bn


def is_close(a, b, tol=1e-5):
    return ((a - b).abs().mean() / b.abs().mean()).item() < tol


def check_correctness(f1, f2, device):
    # Initialize inputs
    x, W1, W2, b1, b2, wn, bn = create_input(device, dtype=torch.float32, grad=False)

    # Run forward
    y1 = f1(x, W1, W2, b1, b2, wn, bn)
    y2 = f2(x, W1.t(), W2.t(), b1, b2, wn, bn)

    # Check correctnessq
    if not is_close(y1, y2):
        print("Forward failed")


def speed(func, its=100, warmup=100):
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(its):
        func()
    torch.cuda.synchronize()
    time_a = time.time() - start
    time_a /= its
    return time_a * 1000


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(5, 12)],
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=[
            "torch",
            "torch-compile",
            "triton",
        ],  # Possible values for `line_arg`.
        line_names=["Torch", "Torch-compile", "Triton"],  # Label name for the lines.
        plot_name="performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(size, provider):
    x, W1, W2, b1, b2, wn, bn = create_input(
        device="cuda",
        dtype=torch.bfloat16,
        grad=False,
        size=size,
    )
    if provider == "triton":
        ms = speed(lambda: fused(x, W1, W2, b1, b2, wn, bn))
    if provider == "torch":
        W1 = W1.t().contiguous()
        W2 = W2.t().contiguous()
        ms = speed(lambda: unfused(x, W1, W2, b1, b2, wn, bn))
    if provider == "torch-compile":
        W1 = W1.t().contiguous()
        W2 = W2.t().contiguous()
        ms = speed(lambda: compiled(x, W1, W2, b1, b2, wn, bn))

    return ms


def clear_gradients(*args):
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.grad is not None:
            arg.grad = None


def clear_memory(device):
    if device.type == "cuda":
        torch._C._cuda_clearCublasWorkspaces()
        torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "mps":
        gc.collect()
        torch.mps.empty_cache()
    else:
        gc.collect()


def peak_memory(f, *args, device=None):
    for _ in range(10):
        clear_memory(device)
        clear_gradients(*args)
        f(*args)
        if device.type == "cuda":
            torch.cuda.synchronize()
            memory = torch.cuda.max_memory_allocated(device)
        elif device.type == "mps":
            torch.mps.synchronize()
            memory = torch.mps.current_allocated_memory()
        else:
            memory = 0
    return memory


def memory_triton(f, device=None):
    clear_memory(device)
    x, W1, W2, b1, b2, wn, bn = create_input(
        device=device, dtype=torch.bfloat16, grad=False
    )
    print("Current memory: ", (
        torch.cuda.memory_allocated(device) / (1024**3)
        if device.type == "cuda"
        else torch.mps.current_allocated_memory() / (1024**3)
        if device.type == "mps"
        else 0
    ))
    memory = peak_memory(f, x, W1, W2, b1, b2, wn, bn, device=device)
    print("Peak memory: ", memory / (1024**3))
    return memory


def memory_baselines(f, device=None):
    clear_memory(device)
    x, W1, W2, b1, b2, wn, bn = create_input(
        device=device, dtype=torch.bfloat16, grad=False
    )
    W1 = W1.t().contiguous()
    W2 = W2.t().contiguous()
    print("Current memory: ", (
        torch.cuda.memory_allocated(device) / (1024**3)
        if device.type == "cuda"
        else torch.mps.current_allocated_memory() / (1024**3)
        if device.type == "mps"
        else 0
    ))
    memory = peak_memory(f, x, W1, W2, b1, b2, wn, bn, device=device)
    print("Peak memory: ", memory / (1024**3))
    return memory


def test():
    # Setup
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Check correctness
    check_correctness(fused, unfused, device=device)

    # Compute performance
    print("")
    print("Performance")
    if device.type == "cuda":
        benchmark.run(print_data=True, show_plots=False)
    else:
        print("Benchmarking only supported on CUDA.")
    print("")

    # Compute memory
    memory_fused = memory_triton(fused, device=device)
    memory_unfused = memory_baselines(unfused, device=device)
    memory_compile = memory_baselines(compiled, device=device)

    print("")
    print("Memory savings against unfused: ", memory_unfused / memory_fused if memory_fused else "N/A")
    print("Memory savings against compile: ", memory_compile / memory_fused if memory_fused else "N/A")
    print("")


if __name__ == "__main__":
    test()
