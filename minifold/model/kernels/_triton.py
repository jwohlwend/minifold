"""Optional triton import: triton is only available on CUDA (Linux) builds.

When it is missing (e.g. macOS/MPS), `triton` and `tl` are inert placeholders so
the kernel modules still import; the kernels themselves are only launched on CUDA
and fall back to plain torch ops otherwise.
"""

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

    class _Placeholder:
        def __getattr__(self, name):
            return self

        def __call__(self, *args, **kwargs):
            return self

    triton = tl = _Placeholder()
