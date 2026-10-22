import torch
import torch.nn.functional as F
from minifold.model.kernels._triton import triton, tl
import gc


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
    # Get program id
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
        shape=(C2, C1),
        strides=(C1, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_C2, C1),
        order=(1, 0),
    )

    b2_block_ptr = tl.make_block_ptr(
        base=b2_ptr,
        shape=(C1,),
        strides=(1,),
        offsets=(0,),
        block_shape=(C1,),
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
        shape=(M, C1),
        strides=(C1, 1),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, C1),
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

    # Compute output
    accum_x = tl.zeros((BLOCK_SIZE_M, C1), dtype=tl.float32)
    for _ in range(0, tl.cdiv(C2, BLOCK_SIZE_C2)):
        # Load weights
        w = tl.load(w1_block_ptr).to(dtype)
        b = tl.load(b1_block_ptr).to(dtype)

        # Compute hidden
        h = tl.dot(x, w, allow_tf32=False)
        h += b[None, :]
        h = tl.where(h > 0, h, 0.)
        h = h.to(dtype)

        # Update x
        w = tl.load(w2_block_ptr).to(dtype)
        accum_x += tl.dot(h, w, allow_tf32=False)

        # Advance pointers
        w1_block_ptr = tl.advance(w1_block_ptr, (0, BLOCK_SIZE_C2))
        w2_block_ptr = tl.advance(w2_block_ptr, (BLOCK_SIZE_C2, 0))
        b1_block_ptr = tl.advance(b1_block_ptr, (BLOCK_SIZE_C2,))

    # Store output
    b = tl.load(b2_block_ptr)
    accum_x += b[None, :]
    accum_x = accum_x.to(dtype)
    tl.store(o_block_ptr, accum_x)


def mlp_kernel(X, W1, W2, b1, b2, wn, bn):
    if X.device.type == "cuda":
        B, N1, N2, C1 = X.shape
        M = B * N1 * N2
        C2 = W1.shape[1]

        X = X.contiguous()
        O = torch.empty_like(X)

        BLOCK_SIZE_M = 128
        BLOCK_SIZE_C2 = 32

        assert M > BLOCK_SIZE_M
        assert M % BLOCK_SIZE_M == 0
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
        x = F.linear(x, W1, b1)
        x.relu_()
        x = F.linear(x, W2, b2)
        return x


#---------------- TESTS ----------------#

def fused(x, W1, W2, b1, b2, wn, bn):
    x = mlp_kernel(x, W1, W2, b1, b2, wn, bn)
    return x


def unfused(x, W1, W2, b1, b2, wn, bn):
    x = F.layer_norm(x, x.shape[-1:], weight=wn, bias=bn, eps=1e-5)
    x = F.linear(x, W1, b1)
    x.relu_()
    x = F.linear(x, W2, b2)
    return x


@torch.compile
def compiled(x, W1, W2, b1, b2, wn, bn):
    x = F.layer_norm(x, x.shape[-1:], weight=wn, bias=bn, eps=1e-5)
    x = F.linear(x, W1, b1)
    x.relu_()
    x = F.linear(x, W2, b2)
    return x


def create_input(device, dtype=torch.float32, grad=False, size=256):
    B = 1
    C = 128
    H = C * 4
    N = size

    x = 0.1 * torch.randn((B, N, N, C), device=device, dtype=dtype)
    W1 = 0.1 * torch.randn((C, H), device=device, dtype=dtype)
    W2 = 0.1 * torch.randn((H, C), device=device, dtype=dtype)
    b1 = 0.1 * torch.randn((H,), device=device, dtype=dtype)
    b2 = 0.1 * torch.randn((C,), device=device, dtype=dtype)
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

    # Check correctness
    if not is_close(y1, y2):
        print("Forward failed")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(5, 13)],
        line_arg="provider",
        line_vals=["triton", "torch-compile", "torch"],
        line_names=["Triton", "Torch-compile", "Torch"],
        plot_name="performance",
        args={},
    )
)
def benchmark(size, provider, device="cuda"):
    x, W1, W2, b1, b2, wn, bn = create_input(
        device, dtype=torch.bfloat16, grad=False, size=size
    )

    if provider == "triton":
        if device == "cuda":
            ms = triton.testing.do_bench(lambda: fused(x, W1, W2, b1, b2, wn, bn))
        else:
            ms = float("nan")
    if provider == "torch":
        W1 = W1.t().contiguous()
        W2 = W2.t().contiguous()
        ms = triton.testing.do_bench(lambda: unfused(x, W1, W2, b1, b2, wn, bn))
    if provider == "torch-compile":
        W1 = W1.t().contiguous()
        W2 = W2.t().contiguous()
        ms = triton.testing.do_bench(lambda: compiled(x, W1, W2, b1, b2, wn, bn))

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
    x, W1, W2, b1, b2, wn, bn = create_input(device, dtype=torch.bfloat16, grad=False)
    if device.type == "cuda":
        print("Current memory: ", torch.cuda.memory_allocated(device) / (1024**3))
    elif device.type == "mps":
        print("Current memory: ", torch.mps.current_allocated_memory() / (1024**3))
    else:
        print("Current memory: 0")
    memory = peak_memory(f, x, W1, W2, b1, b2, wn, bn, device=device)
    print("Peak memory: ", memory / (1024**3))
    return memory


def memory_baseline(f, device=None):
    clear_memory(device)
    x, W1, W2, b1, b2, wn, bn = create_input(device, dtype=torch.bfloat16, grad=False)
    W1 = W1.t().contiguous()
    W2 = W2.t().contiguous()
    if device.type == "cuda":
        print("Current memory: ", torch.cuda.memory_allocated(device) / (1024**3))
    elif device.type == "mps":
        print("Current memory: ", torch.mps.current_allocated_memory() / (1024**3))
    else:
        print("Current memory: 0")
    memory = peak_memory(f, x, W1, W2, b1, b2, wn, bn, device=device)
    print("Peak memory: ", memory / (1024**3))
    return memory


def test():
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    check_correctness(fused, unfused, device=device)

    print("")
    print("Performance")
    if device.type == "cuda":
        benchmark.run(print_data=True, show_plots=False, device="cuda")
    elif device.type == "mps":
        benchmark.run(print_data=True, show_plots=False, device="mps")
    else:
        benchmark.run(print_data=True, show_plots=False, device="cpu")
    print("")

    if device.type == "cuda":
        memory_fused = memory_triton(fused, device=device)
        memory_unfused = memory_baseline(unfused, device=device)
        memory_compile = memory_baseline(compiled, device=device)
        print("")
        print("Memory savings against unfused: ", memory_unfused / memory_fused)
        print("Memory savings against compile: ", memory_compile / memory_fused)
        print("")
    else:
        print("Memory profiling only supported on CUDA.")


if __name__ == "__main__":
    test()
