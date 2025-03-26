import pathlib
import warnings

import torch.utils.cpp_extension


srcpath = pathlib.Path(__file__).parent.absolute()

_geglu_ext = None


def get_geglu_ext():
    global _geglu_ext
    if _geglu_ext is None:
        _geglu_ext = torch.utils.cpp_extension.load(
            "nemo_stable_diffusion_fast_geglu_ext",
            sources=[
                srcpath / "geglu.cpp",
                srcpath / "geglu_cuda.cu",
            ],
            extra_cuda_cflags=[
                "-O2",
                "--use_fast_math",
                "--ftz=false",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
            ],
            verbose=True,
        )
    return _geglu_ext


@torch.library.custom_op("fast_geglu::geglu_fprop", mutates_args=())
def geglu_fprop(x_and_gate: torch.Tensor) -> torch.Tensor:
    assert x_and_gate.is_cuda
    assert x_and_gate.dtype == torch.float16
    assert x_and_gate.is_contiguous()
    geglu_ext = get_geglu_ext()
    dim_batch = x_and_gate.shape[:-1]
    dim_last = x_and_gate.shape[-1] // 2
    out = torch.empty(dim_batch + (dim_last,), dtype=x_and_gate.dtype, device=x_and_gate.device)
    geglu_ext.geglu(
        out.data_ptr(), x_and_gate.data_ptr(), dim_batch.numel(), dim_last, torch.cuda.current_stream().cuda_stream
    )
    return out


@geglu_fprop.register_fake
def geglu_fprop_fake(x_and_gate: torch.Tensor) -> torch.Tensor:
    assert x_and_gate.is_cuda
    assert x_and_gate.dtype == torch.float16
    assert x_and_gate.is_contiguous()
    dim_batch = x_and_gate.shape[:-1]
    dim_last = x_and_gate.shape[-1] // 2
    out = torch.empty(dim_batch + (dim_last,), dtype=x_and_gate.dtype, device=x_and_gate.device)
    return out


@torch.library.custom_op("fast_geglu::geglu_bprop", mutates_args=())
def geglu_bprop(grad_out: torch.Tensor, x_and_gate: torch.Tensor) -> torch.Tensor:
    assert grad_out.is_cuda
    assert grad_out.dtype == torch.float16
    assert grad_out.is_contiguous()
    assert x_and_gate.device == grad_out.device
    assert x_and_gate.dtype == torch.float16
    assert x_and_gate.is_contiguous()
    geglu_ext = get_geglu_ext()
    dim_batch = x_and_gate.shape[:-1]
    dim_last = x_and_gate.shape[-1] // 2
    grad_x_and_gate = torch.empty_like(x_and_gate)
    geglu_ext.geglu_bwd(
        grad_x_and_gate.data_ptr(),
        grad_out.data_ptr(),
        x_and_gate.data_ptr(),
        dim_batch.numel(),
        dim_last,
        torch.cuda.current_stream().cuda_stream,
    )
    return grad_x_and_gate


@geglu_bprop.register_fake
def geglu_bprop_fake(grad_out: torch.Tensor, x_and_gate: torch.Tensor) -> torch.Tensor:
    assert grad_out.is_cuda
    assert grad_out.dtype == torch.float16
    assert grad_out.is_contiguous()
    assert x_and_gate.device == grad_out.device
    assert x_and_gate.dtype == torch.float16
    assert x_and_gate.is_contiguous()
    grad_x_and_gate = torch.empty_like(x_and_gate)
    return grad_x_and_gate


def backward(ctx, grad_out):
    (x_and_gate,) = ctx.saved_tensors
    grad_x_and_gate = geglu_bprop(grad_out, x_and_gate)
    return grad_x_and_gate


def setup_context(ctx, inputs, output):
    (x_and_gate,) = inputs
    ctx.save_for_backward(x_and_gate)


geglu_fprop.register_autograd(backward, setup_context=setup_context)


def geglu(x_and_gate):
    if (
        x_and_gate.is_cuda
        and x_and_gate.dtype == torch.float16
        and x_and_gate.shape[-1] // 2 in [1280, 2560, 5120]
        and x_and_gate.is_contiguous()
    ):
        return geglu_fprop(x_and_gate)
    else:
        warnings.warn("Fast GeGLU is not applied. Falling back to the PyTorch.", UserWarning)
        x, gate = x_and_gate.chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gate)


__all__ = ["geglu"]
