#include "geglu.hpp"

#include <cstdio>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cuda_fp16.h>


#define CUDA_CHECK(call)                                                    \
do {                                                                        \
    cudaError_t err_ = call;                                                \
    if (err_ != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",        \
                __FILE__, __LINE__, err_, cudaGetErrorString(err_), #call); \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
} while (0)

namespace {

__device__ float exp_auto(float x) {
    return ::expf(x);
}

[[maybe_unused]] __device__ double exp_auto(double x) {
    return ::exp(x);
}

template<typename opmath_t>
__device__ __inline__ opmath_t gelu(opmath_t x) {
    // Refer to the PyTorch implementation:
    //   https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L37-L38
    constexpr opmath_t kAlpha = M_SQRT1_2;
    return x * opmath_t(0.5) * (opmath_t(1) + ::erf(x * kAlpha));
}

template<typename opmath_t>
__device__ __inline__ opmath_t gelu_bwd(opmath_t dy, opmath_t x) {
    // Refer to the PyTorch implementation:
    //   https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L74-L82
    constexpr opmath_t kBeta = M_2_SQRTPI * M_SQRT1_2 * opmath_t(0.5);
    constexpr opmath_t kAlpha = M_SQRT1_2;
    const opmath_t cdf =
        opmath_t(0.5) * (opmath_t(1) + ::erf(x * kAlpha));
    const opmath_t pdf =
        exp_auto(
            opmath_t(-0.5) * x * x) *
        kBeta;
    return dy * (cdf + x * pdf);
}

}

template<typename scalar_t, typename opmath_t, int BLOCK_DIM_X, int DIM_LAST, int VEC_ELEMS, int FOR_LOOP>
__global__ __launch_bounds__(BLOCK_DIM_X) void geglu_kernel(scalar_t *out, scalar_t const *x_and_gate) {
    static_assert(DIM_LAST % (BLOCK_DIM_X * VEC_ELEMS) == 0, "cannot decide CHUNKS_PER_ROW");
    constexpr int CHUNKS_PER_ROW = DIM_LAST / (BLOCK_DIM_X * VEC_ELEMS);
    struct alignas(sizeof(scalar_t) * VEC_ELEMS) U {
        scalar_t data[VEC_ELEMS];
    };
    U ux[FOR_LOOP];
    U ugate[FOR_LOOP];
    U uout[FOR_LOOP];
    for (int k = 0; k < FOR_LOOP; k++) {
        int idxN = (blockIdx.x * FOR_LOOP + k) / CHUNKS_PER_ROW;
        int idxR = ((blockIdx.x * FOR_LOOP + k) % CHUNKS_PER_ROW * BLOCK_DIM_X + threadIdx.x) * VEC_ELEMS;
        ux[k]    = *reinterpret_cast<U const *>(&x_and_gate[(idxN * 2 + 0) * (int64_t)DIM_LAST + idxR]);
        ugate[k] = *reinterpret_cast<U const *>(&x_and_gate[(idxN * 2 + 1) * (int64_t)DIM_LAST + idxR]);
    }
    for (int k = 0; k < FOR_LOOP; k++) {
        for (int i = 0; i < VEC_ELEMS; i++) {
            constexpr opmath_t kAlpha = M_SQRT1_2;
            opmath_t gelu_out = gelu(static_cast<opmath_t>(ugate[k].data[i]));
            uout[k].data[i] = static_cast<scalar_t>(static_cast<opmath_t>(ux[k].data[i]) * gelu_out);
        }
    }
    for (int k = 0; k < FOR_LOOP; k++) {
        int idxN = (blockIdx.x * FOR_LOOP + k) / CHUNKS_PER_ROW;
        int idxR = ((blockIdx.x * FOR_LOOP + k) % CHUNKS_PER_ROW * BLOCK_DIM_X + threadIdx.x) * VEC_ELEMS;
        *reinterpret_cast<U *>(&out[idxN * (int64_t)DIM_LAST + idxR]) = uout[k];
    }
}

template<typename scalar_t, typename opmath_t, int BLOCK_DIM_X, int DIM_LAST, int VEC_ELEMS, int FOR_LOOP>
__global__ __launch_bounds__(BLOCK_DIM_X) void geglu_bwd_kernel(scalar_t *grad_x_and_gate, scalar_t const *grad_out, scalar_t const *x_and_gate) {
    static_assert(DIM_LAST % (BLOCK_DIM_X * VEC_ELEMS) == 0, "DIM_LAST must be chunked to multiple IO blocks");
    constexpr int CHUNKS_PER_ROW = DIM_LAST / (BLOCK_DIM_X * VEC_ELEMS);
    struct alignas(sizeof(scalar_t) * VEC_ELEMS) U {
        scalar_t data[VEC_ELEMS];
    };
    U ugrad_out[FOR_LOOP];
    U ux[FOR_LOOP];
    U ugate[FOR_LOOP];
    U ugrad_x[FOR_LOOP];
    U ugrad_gate[FOR_LOOP];
    for (int k = 0; k < FOR_LOOP; k++) {
        int idxN = (blockIdx.x * FOR_LOOP + k) / CHUNKS_PER_ROW;
        int idxR = ((blockIdx.x * FOR_LOOP + k) % CHUNKS_PER_ROW * BLOCK_DIM_X + threadIdx.x) * VEC_ELEMS;
        ugrad_out[k] = *reinterpret_cast<U const *>(&grad_out[idxN * (int64_t)DIM_LAST + idxR]);
        ux[k]    = *reinterpret_cast<U const *>(&x_and_gate[(idxN * 2 + 0) * (int64_t)DIM_LAST + idxR]);
        ugate[k] = *reinterpret_cast<U const *>(&x_and_gate[(idxN * 2 + 1) * (int64_t)DIM_LAST + idxR]);
    }
    for (int k = 0; k < FOR_LOOP; k++) {
        for (int i = 0; i < VEC_ELEMS; i++) {
            ugrad_x[k].data[i] = static_cast<scalar_t>(gelu(static_cast<opmath_t>(ugate[k].data[i])) * static_cast<opmath_t>(ugrad_out[k].data[i]));
            opmath_t grad_gelu_out = static_cast<opmath_t>(ux[k].data[i]) * static_cast<opmath_t>(ugrad_out[k].data[i]);
            opmath_t grad_gate = gelu_bwd(grad_gelu_out, static_cast<opmath_t>(ugate[k].data[i]));
            ugrad_gate[k].data[i] = static_cast<scalar_t>(grad_gate);
        }
    }
    for (int k = 0; k < FOR_LOOP; k++) {
        int idxN = (blockIdx.x * FOR_LOOP + k) / CHUNKS_PER_ROW;
        int idxR = ((blockIdx.x * FOR_LOOP + k) % CHUNKS_PER_ROW * BLOCK_DIM_X + threadIdx.x) * VEC_ELEMS;
        *reinterpret_cast<U *>(&grad_x_and_gate[(idxN * 2 + 0) * (int64_t)DIM_LAST + idxR]) = ugrad_x[k];
        *reinterpret_cast<U *>(&grad_x_and_gate[(idxN * 2 + 1) * (int64_t)DIM_LAST + idxR]) = ugrad_gate[k];
    }
}

#define DISPATCH_DIM_LAST(VALUE, CONST_NAME, ...) [&] { \
    if (VALUE == 1280) { constexpr int CONST_NAME = 1280; return __VA_ARGS__(); } \
    if (VALUE == 2560) { constexpr int CONST_NAME = 2560; return __VA_ARGS__(); } \
    if (VALUE == 5120) { constexpr int CONST_NAME = 5120; return __VA_ARGS__(); } \
    throw std::invalid_argument("DISPATCH_DIM_LAST " + std::to_string(VALUE)); \
    }()

#define DISPATCH_FOR_LOOP(VALUE, CONST_NAME, ...) [&] { \
    if (VALUE == 1) { constexpr int CONST_NAME = 1; return __VA_ARGS__(); } \
    if (VALUE == 2) { constexpr int CONST_NAME = 2; return __VA_ARGS__(); } \
    throw std::invalid_argument("DISPATCH_FOR_LOOP " + std::to_string(VALUE)); \
    }()

void geglu_cuda(intptr_t out_, intptr_t x_and_gate_, int64_t n, int dim_last, intptr_t stream_) {
    using scalar_t = half;
    scalar_t *out = reinterpret_cast<scalar_t *>(out_);
    scalar_t const *x_and_gate = reinterpret_cast<scalar_t const *>(x_and_gate_);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_);
    using opmath_t = float;
    constexpr int VEC_ELEMS = 8;
    constexpr int BLOCK_DIM_X = 160;
    int for_loop = 2;
    while (for_loop > 0 && n * dim_last % (BLOCK_DIM_X * VEC_ELEMS * for_loop) != 0) {
        for_loop /= 2;
    }
    if (for_loop == 0) {
        throw std::invalid_argument("cannot determine grid_dim");
    }
    dim3 grid_dim(n * dim_last / (BLOCK_DIM_X * VEC_ELEMS * for_loop));
    DISPATCH_FOR_LOOP(for_loop, FOR_LOOP, [&] {
        DISPATCH_DIM_LAST(dim_last, DIM_LAST, [&] {
            geglu_kernel<scalar_t, opmath_t, BLOCK_DIM_X, DIM_LAST, VEC_ELEMS, FOR_LOOP><<<grid_dim, BLOCK_DIM_X, 0, stream>>>(out, x_and_gate);
        });
    });
    CUDA_CHECK(cudaPeekAtLastError());
}

void geglu_bwd_cuda(intptr_t grad_x_and_gate_, intptr_t grad_out_, intptr_t x_and_gate_, int64_t n, int dim_last, intptr_t stream_) {
    using scalar_t = half;
    scalar_t *grad_x_and_gate = reinterpret_cast<scalar_t *>(grad_x_and_gate_);
    scalar_t const *grad_out = reinterpret_cast<scalar_t const *>(grad_out_);
    scalar_t const *x_and_gate = reinterpret_cast<scalar_t const *>(x_and_gate_);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_);
    using opmath_t = float;
    constexpr int VEC_ELEMS = 8;
    constexpr int BLOCK_DIM_X = 160;
    int for_loop = 1;
    while (for_loop > 0 && n * dim_last % (BLOCK_DIM_X * VEC_ELEMS * for_loop) != 0) {
        for_loop /= 2;
    }
    if (for_loop == 0) {
        throw std::invalid_argument("cannot determine grid_dim");
    }
    dim3 grid_dim(n * dim_last / (BLOCK_DIM_X * VEC_ELEMS * for_loop));
    DISPATCH_FOR_LOOP(for_loop, FOR_LOOP, [&] {
        DISPATCH_DIM_LAST(dim_last, DIM_LAST, [&] {
            geglu_bwd_kernel<scalar_t, opmath_t, BLOCK_DIM_X, DIM_LAST, VEC_ELEMS, FOR_LOOP><<<grid_dim, BLOCK_DIM_X, 0, stream>>>(grad_x_and_gate, grad_out, x_and_gate);
        });
    });
    CUDA_CHECK(cudaPeekAtLastError());
}
