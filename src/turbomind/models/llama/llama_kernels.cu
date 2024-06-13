// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/attention/array_ops.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"
#include "src/turbomind/kernels/reduce_kernel_utils.cuh"
#include "src/turbomind/macro.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/cuda_type_utils.cuh"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/dispatch.h"
#include "src/turbomind/utils/logger.h"
#include <algorithm>
#include <cstdint>
#include <cub/block/block_reduce.cuh>
#include <type_traits>
#include <utility>

namespace turbomind {

// fp16, bf16
// n is divided by 2 for this impl
template<typename T>
__global__ void rootMeanSquareNorm(T* out, const T* input, const T* scale, float eps, int m, int n)
{
    using T2 = typename TypeConverter<T>::Type;
    __shared__ float s_inv_mean;
    float            mean = 0.f;

    T2*       out_ptr   = (T2*)out;
    const T2* input_ptr = (const T2*)input;
    const T2* scale_ptr = (const T2*)scale;

    for (uint idx = threadIdx.x; idx < n; idx += blockDim.x) {
        float2 tmp2 = cuda_cast<float2>(input_ptr[blockIdx.x * n + idx]);
        mean += tmp2.x * tmp2.x;
        mean += tmp2.y * tmp2.y;
    }

    mean = blockReduceSum<float>(mean);
    if (threadIdx.x == 0) {
        s_inv_mean = rsqrt(.5f * mean / (float)n + eps);
    }
    __syncthreads();

    for (uint idx = threadIdx.x; idx < n; idx += blockDim.x) {
        float2 tmp2                   = cuda_cast<float2>(input_ptr[blockIdx.x * n + idx]);
        float2 sca2                   = cuda_cast<float2>(scale_ptr[idx]);
        tmp2.x                        = tmp2.x * s_inv_mean * sca2.x;
        tmp2.y                        = tmp2.y * s_inv_mean * sca2.y;
        out_ptr[blockIdx.x * n + idx] = cuda_cast<T2>(tmp2);
    }
}

template<>
__global__ void rootMeanSquareNorm(float* out, const float* input, const float* scale, float eps, int m, int n)
{
    __shared__ float s_inv_mean;
    float            mean = 0.f;

    for (uint idx = threadIdx.x; idx < n; idx += blockDim.x) {
        float tmp = input[blockIdx.x * n + idx];
        mean += tmp * tmp;
    }

    mean = blockReduceSum<float>(mean);
    if (threadIdx.x == 0) {
        s_inv_mean = rsqrt(mean / static_cast<float>(n) + eps);
    }
    __syncthreads();

    for (uint idx = threadIdx.x; idx < n; idx += blockDim.x) {
        float tmp                 = input[blockIdx.x * n + idx];
        out[blockIdx.x * n + idx] = tmp * s_inv_mean * scale[idx];
    }
}

template<typename T>
void invokeRootMeanSquareNorm(T* out, const T* input, const T* scale, float eps, int m, int n, cudaStream_t stream)
{
    if (sizeof(T) == 2) {
        FT_CHECK(n % 2 == 0);
        n /= 2;
    }
    dim3 grid(m);
    dim3 block(std::min(n, 1024));
    rootMeanSquareNorm<<<grid, block, 0, stream>>>(out, input, scale, eps, m, n);
}

template void invokeRootMeanSquareNorm(float*, const float*, const float*, float, int, int, cudaStream_t);
template void invokeRootMeanSquareNorm(half*, const half*, const half*, float, int, int, cudaStream_t);
#ifdef ENABLE_BF16
template void
invokeRootMeanSquareNorm(__nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, float, int, int, cudaStream_t);
#endif

// #ifdef ENABLE_BF16

// template void invokeRootMeanSquareNorm(__nv_bfloat16*, const __nv_bfloat16*, float, int, int, cudaStream_t);

// #endif

template<typename T, typename T0>
__device__ T saturate_cast(T0 x)
{
    return x;
}

template<>
__device__ half saturate_cast<half, float>(float x)
{
    return (x > 64512.f || x < -64512.f) ? (x > 0.f ? 64512.f : -64512.f) : x;
}

template<typename T>
__global__ void addResidual(T* out, const T* in, size_t n)
{
    auto idx = threadIdx.x + (size_t)blockIdx.x * blockDim.x;
    if (idx < n) {
        out[idx] = static_cast<T>(static_cast<float>(out[idx]) + static_cast<float>(in[idx]));
    }
}

template<typename T>
void invokeAddResidual(T* out, const T* in, int m, int n, cudaStream_t stream)
{
    auto total = static_cast<size_t>(m) * n;
    dim3 block(std::min((unsigned long)total, 1024UL));
    dim3 grid((total + block.x - 1) / block.x);

    addResidual<<<grid, block, 0, stream>>>(out, in, total);
}

template void invokeAddResidual(float*, const float*, int, int, cudaStream_t);
template void invokeAddResidual(half*, const half*, int, int, cudaStream_t);

// ids [seq_len, batch_size]
// input_ids [batch_size, max_input_len]
__global__ void
fixInputIds(int* ids, const int* input_ids, const int* input_lengths, int batch_size, int seq_len, int max_input_len)
{
    int seq_id   = threadIdx.x;
    int batch_id = blockIdx.x;
    for (; seq_id < input_lengths[batch_id]; seq_id += blockDim.x) {
        ids[seq_id * batch_size + batch_id] = input_ids[batch_id * max_input_len + seq_id];
    }
}

void invokeFixInputIds(int*         ids,
                       const int*   input_ids,
                       const int*   input_lengths,
                       int          batch_size,
                       int          seq_len,
                       int          max_input_len,
                       cudaStream_t st)
{
    dim3 block(std::min(1024, max_input_len));
    dim3 grid(batch_size);
    fixInputIds<<<grid, block, 0, st>>>(ids, input_ids, input_lengths, batch_size, seq_len, max_input_len);
}

template<typename T>
__global__ void sliceCausalMask(T* mask, int seq_len, int key_len, int step)
{
    mask += (size_t)blockIdx.x * seq_len * key_len;
    for (int i = threadIdx.x; i < seq_len * key_len; i += blockDim.x) {
        int row = i / key_len;
        int col = i % key_len;
        if (col <= row + step) {
            mask[i] = static_cast<T>(1.f);
        }
        else {
            mask[i] = static_cast<T>(0.f);
        }
    }
}

// [step: step+Q, :] of the K*K causal mask
template<typename T>
void invokeSliceCausalMask(T* mask, int seq_len, int key_len, int step, int batch_size, cudaStream_t stream)
{
    FT_CHECK(step == key_len - seq_len);
    sliceCausalMask<<<batch_size, 256, 0, stream>>>(mask, seq_len, key_len, step);
}

template void invokeSliceCausalMask(half*, int, int, int, int, cudaStream_t);
template void invokeSliceCausalMask(float*, int, int, int, int, cudaStream_t);

// mask [bsz, max_q_len, max_k_len]

template<typename T>
__global__ void createCausalMasks(T* mask, const int* q_lens, const int* k_lens, int max_q_len, int max_k_len)
{
    const auto q_len = q_lens ? q_lens[blockIdx.x] : max_q_len;
    const auto k_len = k_lens ? k_lens[blockIdx.x] : max_k_len;
    mask += blockIdx.x * max_q_len * max_k_len;
    for (int i = threadIdx.x; i < max_q_len * max_k_len; i += blockDim.x) {
        const int q        = i / max_k_len;  // [0, max_q_len)
        const int k        = i % max_k_len;  // [0, max_k_len)
        bool      is_valid = q < q_len && k < k_len && k <= q + (k_len - q_len);
        mask[i]            = static_cast<T>(is_valid);
    }
}

template<typename T>
void invokeCreateCausalMasks(
    T* mask, const int* q_lens, const int* k_lens, int max_q_len, int max_k_len, int batch_size, cudaStream_t stream)
{
    createCausalMasks<<<batch_size, 512, 0, stream>>>(mask, q_lens, k_lens, max_q_len, max_k_len);
}

template void invokeCreateCausalMasks(float* mask, const int*, const int*, int, int, int, cudaStream_t);
template void invokeCreateCausalMasks(half* mask, const int*, const int*, int, int, int, cudaStream_t);
#ifdef ENABLE_BF16
template<>
__global__ void createCausalMasks<__nv_bfloat16>(
    __nv_bfloat16* mask, const int* q_lens, const int* k_lens, int max_q_len, int max_k_len)
{
    const auto q_len = q_lens[blockIdx.x];
    const auto k_len = k_lens[blockIdx.x];
    mask += blockIdx.x * max_q_len * max_k_len;
    for (int i = threadIdx.x; i < max_q_len * max_k_len; i += blockDim.x) {
        const int q        = i / max_k_len;  // [0, max_q_len)
        const int k        = i % max_k_len;  // [0, max_k_len)
        bool      is_valid = q < q_len && k < k_len && k <= q + (k_len - q_len);
        mask[i]            = static_cast<__nv_bfloat16>(float(is_valid));
    }
}
template void invokeCreateCausalMasks(__nv_bfloat16* mask, const int*, const int*, int, int, int, cudaStream_t);
#endif

namespace {

template<class Kernel, class Params>
__global__ void KernelWrapper(Params params)
{
    Kernel{}(params);
};

}  // namespace

__global__ void gatherOutput(int*       output_ids,
                             const int* ids,
                             const int* context_length,
                             int        max_context_len,
                             int        max_gen_step,
                             int        max_output_len,
                             int        batch_size)
{
    const int batch_id    = blockIdx.x;
    const int context_len = context_length[batch_id];
    output_ids += batch_id * max_output_len;
    for (int src_idx = threadIdx.x; src_idx < max_gen_step; src_idx += blockDim.x) {
        // skip padding for src
        if (context_len <= src_idx && src_idx < max_context_len) {
            continue;
        }
        // skip padding for dst
        const int dst_idx = src_idx < context_len ? src_idx : src_idx - (max_context_len - context_len);
        if (dst_idx < max_output_len) {
            output_ids[dst_idx] = ids[src_idx * batch_size + batch_id];
        }
    }
}

void invokeGatherOutput(int*         output_ids,
                        const int*   ids,
                        const int*   context_length,
                        int          max_context_len,
                        int          max_gen_step,
                        int          max_output_len,
                        int          batch_size,
                        cudaStream_t stream)
{
    int block_size = 128;
    int grid_size  = batch_size;
    gatherOutput<<<grid_size, block_size, 0, stream>>>(
        output_ids, ids, context_length, max_context_len, max_gen_step, max_output_len, batch_size);
}

__global__ void gatherOutput(int*       output_ids,
                             int*       next_input_ids,
                             const int* ids,
                             const int* last_input_ids,
                             const int* verified_length,
                             const int* context_length,
                             const int* verified_packed_path,
                             int        max_context_len,
                             int        max_gen_step,
                             int        max_output_len,
                             int        stride_len,
                             int        batch_size,
                             int        medusa_head_num)
{
    const int batch_id    = blockIdx.x;
    const int context_len = context_length[batch_id];
    output_ids += batch_id * max_output_len;
    const int verified_len = verified_length ? verified_length[batch_id] : 0;
    if (next_input_ids) {
        next_input_ids += batch_id * max_output_len;
    }
    if (verified_packed_path) {
        verified_packed_path += batch_id * (1 + medusa_head_num);
    }

    for (int src_idx = threadIdx.x; src_idx < max_gen_step; src_idx += blockDim.x) {
        if (context_len <= src_idx && src_idx < max_context_len) {
            continue;
        }
        if (src_idx < max_gen_step - stride_len) {
            const int dst_idx = src_idx < context_len ? src_idx : src_idx - (max_context_len - context_len);
            if (dst_idx < max_output_len) {
                output_ids[dst_idx] = ids[src_idx * batch_size + batch_id];
            }
        }
        else {
            const int input_dst_idx = src_idx - (max_gen_step - stride_len);
            if (next_input_ids) {
                next_input_ids[input_dst_idx] = ids[src_idx * batch_size + batch_id];
            }
            const int last_src_idx = src_idx - (max_gen_step - stride_len);

            if (src_idx != max_gen_step - 1 && last_src_idx == 0) {
                continue;
            }
            if (last_input_ids && last_src_idx <= verified_len) {
                int       true_packed_idx = (verified_packed_path) ? verified_packed_path[last_src_idx] : last_src_idx;
                const int dst_idx         = src_idx - (max_context_len - context_len) - 1;
                if (dst_idx < max_output_len) {
                    output_ids[dst_idx] = last_input_ids[true_packed_idx * batch_size + batch_id];
                }
            }

            if (src_idx == max_gen_step - 1) {
                const int new_src_idx = src_idx - (stride_len - 1);
                const int dst_idx     = src_idx - (max_context_len - context_len) - (stride_len - verified_len) + 1;
                if (dst_idx < max_output_len) {
                    output_ids[dst_idx] = ids[new_src_idx * batch_size + batch_id];
                }
            }
        }
    }
}

void invokeGatherOutput(int*         output_ids,
                        int*         next_input_ids,
                        const int*   ids,
                        const int*   last_input_ids,
                        const int*   verified_length,
                        const int*   context_length,
                        const int*   verified_packed_path,
                        int          max_context_len,
                        int          max_gen_step,
                        int          max_output_len,
                        int          batch_size,
                        int          stride_len,
                        int          medusa_head_num,
                        cudaStream_t stream)
{
    int block_size = 128;
    int grid_size  = batch_size;
    gatherOutput<<<grid_size, block_size, 0, stream>>>(output_ids,
                                                       next_input_ids,
                                                       ids,
                                                       last_input_ids,
                                                       verified_length,
                                                       context_length,
                                                       verified_packed_path,
                                                       max_context_len,
                                                       max_gen_step,
                                                       max_output_len,
                                                       stride_len,
                                                       batch_size,
                                                       medusa_head_num);
}

__global__ void updateOutput(int**      request_output_ids_ptrs,
                             int**      request_seqlen_ptrs,
                             const int* output_ids,
                             const int* sequence_lengths,
                             const int* request_output_ids_lens,
                             int        max_session_len,
                             bool       token_generated)
{
    const int batch_id = blockIdx.x;

    auto request_output_ids = request_output_ids_ptrs[batch_id];
    auto request_seqlen     = request_seqlen_ptrs[batch_id];

    output_ids += max_session_len * batch_id;

    const int seqlen     = sequence_lengths[batch_id] + (int)token_generated;
    const int output_len = min(seqlen, request_output_ids_lens[batch_id]);

    for (int i = threadIdx.x; i < output_len; i += blockDim.x) {
        request_output_ids[i] = output_ids[i];
    }

    *request_seqlen = seqlen;
}

void invokeUpdateOutput(int**        request_output_ids_ptrs,
                        int**        request_seqlen_ptrs,
                        const int*   output_ids,
                        const int*   sequence_lengths,
                        const int*   request_output_ids_lens,
                        int          max_session_len,
                        bool         token_generated,
                        int          batch_size,
                        cudaStream_t stream)
{
    constexpr int block_size = 128;
    const int     grid_size  = batch_size;

    updateOutput<<<grid_size, block_size, 0, stream>>>(request_output_ids_ptrs,
                                                       request_seqlen_ptrs,
                                                       output_ids,
                                                       sequence_lengths,
                                                       request_output_ids_lens,
                                                       max_session_len,
                                                       token_generated);
}

template<int BLOCK_DIM>
__global__ void compactOutputIds(
    int* cu_output_ids, const int* output_ids, const int* sequence_lengths, int session_len, bool token_generated)
{
    typedef cub::BlockReduce<int, BLOCK_DIM>     BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int batch_idx = blockIdx.x;

    int end   = (batch_idx + BLOCK_DIM - 1) / BLOCK_DIM * BLOCK_DIM;  // align to BLOCK_DIM boundary
    int count = 0;
    for (int i = threadIdx.x; i < end; i += blockDim.x) {
        int x = threadIdx.x < batch_idx ? sequence_lengths[threadIdx.x] : 0;
        count += BlockReduce(temp_storage).Sum(x);
        // https://nvlabs.github.io/cub/classcub_1_1_block_reduce.html
        __syncthreads();
    }

    __shared__ int offset;

    if (threadIdx.x == 0) {
        offset = count;
    }

    __syncthreads();

    auto dst = cu_output_ids + offset;

    const int seq_len = sequence_lengths[batch_idx];

    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        dst[i] = output_ids[batch_idx * session_len + i];
    }
}

void invokeCompactOutputIds(int*         cu_output_ids,
                            const int*   output_ids,
                            const int*   sequence_lengths,
                            int          max_session_len,
                            bool         token_generated,
                            int          batch_size,
                            cudaStream_t stream)
{
    constexpr int BLOCK_DIM = 128;
    compactOutputIds<BLOCK_DIM><<<batch_size, BLOCK_DIM, 0, stream>>>(
        cu_output_ids, output_ids, sequence_lengths, max_session_len, token_generated);
}

template<int N, int C>
struct IndexedCopyParam {
    Array<void*, N> src_ptr;
    Array<void*, N> dst_ptr;
    Array<int, N>   stride;
    Array<int, C>   src_idx;
    Array<int, C>   dst_idx;
    int             max_stride;
};

template<class T, int N, int C>
__global__ void indexedCopy(IndexedCopyParam<N, C> param)
{
    const int bi = blockIdx.x;
    const int si = param.src_idx[bi];
    const int di = param.dst_idx[bi];
    for (int i = threadIdx.x; i < param.max_stride; i += blockDim.x) {
        PRAGMA_UNROLL
        for (int k = 0; k < N; ++k) {
            if (i < param.stride[k]) {
                *((T*)param.dst_ptr[k] + param.stride[k] * di + i) =
                    *((const T*)param.src_ptr[k] + param.stride[k] * si + i);
            }
        }
    }
}

template<class T, int N>
void invokeIndexedCopyImpl(void**       h_src_ptr,
                           void**       h_dst_ptr,
                           const int*   h_elem_sz,
                           const int*   h_src_idx,
                           const int*   h_dst_idx,
                           int          count,
                           cudaStream_t st)
{
    dispatch(  // dispatch for num of copy operations
        std::integer_sequence<int, 4, 8, 16, 32, 64, 128, 256>{},
        [&](auto C) { return count <= C; },
        [&](auto C) {
            // maximum parameter size: sm<70: 4kB, sm>=70: 32kB
            static_assert(sizeof(IndexedCopyParam<N, C>) <= 4096);
            IndexedCopyParam<N, C> param{};
            std::copy_n(h_src_ptr, N, param.src_ptr.data());
            std::copy_n(h_dst_ptr, N, param.dst_ptr.data());
            std::transform(h_elem_sz, h_elem_sz + N, param.stride.data(), [](int size) {
                // Basic alignment check
                FT_CHECK_WITH_INFO(size % sizeof(T) == 0, fmtstr("misalignment: %d %% %d", size, (int)sizeof(T)));
                return size / sizeof(T);
            });
            param.max_stride = *std::max_element(param.stride.begin(), param.stride.end());
            auto copy_idx    = [](const int* src, int offset, int n, auto dst) {
                return src ? (void)std::copy_n(src + offset, n, dst) : std::iota(dst, dst + n, offset);
            };
            for (int c = 0; c < count; c += C) {
                int batch_size = std::min(count - c, (int)C);
                copy_idx(h_src_idx, c, batch_size, param.src_idx.data());
                copy_idx(h_dst_idx, c, batch_size, param.dst_idx.data());
                indexedCopy<T><<<batch_size, 128, 0, st>>>(param);
            }
        });
}

void invokeIndexedCopy(void**       h_src_ptr,
                       void**       h_dst_ptr,
                       const int*   h_elem_sz,
                       const int*   h_src_idx,
                       const int*   h_dst_idx,
                       int          count,
                       int          n_copys,
                       cudaStream_t st)
{
    auto success = dispatch(std::integer_sequence<int, 1, 2, 3, 4>{}, [&](auto N) {
        if (N == n_copys) {
            invokeIndexedCopyImpl<uint32_t, N>(h_src_ptr, h_dst_ptr, h_elem_sz, h_src_idx, h_dst_idx, count, st);
            return true;
        }
        return false;
    });
    FT_CHECK(success);
}

__global__ void padLastTokenIds(int* token_ids, const int* context_length, int max_context_len, int batch_size)
{
    for (int bi = threadIdx.x; bi < batch_size; bi += blockDim.x) {
        token_ids[(max_context_len - 1) * batch_size + bi] = token_ids[(context_length[bi] - 1) * batch_size + bi];
    }
}

void invokePadLastTokenIds(
    int* token_ids, const int* context_length, int max_context_len, int batch_size, cudaStream_t stream)
{
    padLastTokenIds<<<1, 512, 0, stream>>>(token_ids, context_length, max_context_len, batch_size);
}

template<typename T>
__global__ void getFeatureOfLastToken(T* output, const T* input, const int* cu_seqlens, int dims)
{
    int bi = blockIdx.x;
    int ti = cu_seqlens[bi + 1] - 1;
    for (int i = threadIdx.x; i < dims; i += blockDim.x) {
        output[dims * bi + i] = input[dims * ti + i];
    }
}

template<typename T>
void invokeGetFeatureOfLastToken(
    T* output, const T* input, const int* cu_seqlens, int dims, int batch_size, cudaStream_t stream)
{
    getFeatureOfLastToken<<<batch_size, 256, 0, stream>>>(output, input, cu_seqlens, dims);
}

template void invokeGetFeatureOfLastToken(half*, const half*, const int*, int, int, cudaStream_t);
template void invokeGetFeatureOfLastToken(float*, const float*, const int*, int, int, cudaStream_t);
#ifdef ENABLE_BF16
template void invokeGetFeatureOfLastToken(__nv_bfloat16*, const __nv_bfloat16*, const int*, int, int, cudaStream_t);
#endif  // ENABLE_BF16

template<class T, int C>
struct BatchedCopyParam {
    Array<T*, C>  src_ptr;
    Array<T*, C>  dst_ptr;
    Array<int, C> size;
    int           count;
};

template<int kThrPerCpy, class T, int C>
__global__ void batchedCopy(BatchedCopyParam<T, C> param)
{
    const int ti = threadIdx.x + blockIdx.x * blockDim.x;
    const int bi = ti / kThrPerCpy;
    if (bi >= param.count) {
        return;
    }
    const T* __restrict__ src = param.src_ptr[bi];
    T* __restrict__ dst       = param.dst_ptr[bi];
    int size                  = param.size[bi];
    for (int i = ti % kThrPerCpy; i < size; i += kThrPerCpy) {
        dst[i] = src[i];
    }
}

// MSVC does not like CUDA kernel launch inside nested lambdas
template<class P>
struct BatchedCopyLauncher {
    int          max_size;
    int          count;
    const P*     params;
    cudaStream_t st;

    template<int S>
    void operator()(std::integral_constant<int, S>) const
    {
        constexpr int threads         = 128;
        constexpr int items_per_block = threads / S;
        const int     blocks          = (count + items_per_block - 1) / items_per_block;
        batchedCopy<S><<<blocks, threads, 0, st>>>(*params);
    }
};

void invokeBatchedCopy(void** src_ptr, void** dst_ptr, int* size, int count, cudaStream_t st)
{
    dispatch(
        std::integer_sequence<int, 1, 8, 32, 128>{},
        [&](auto C) { return count <= C; },
        [&](auto C) {
            using T = uint32_t;
            BatchedCopyParam<T, C> params{};
            // TODO: on CUDA 12.1 and sm_70+ this can be 32K
            static_assert(sizeof(params) <= 4096);
            for (int c = 0; c < count; c += C) {
                const int bsz = std::min<int>(count - c, C);
                params.count  = bsz;
                for (int i = 0; i < bsz; ++i) {
                    params.src_ptr[i] = (T*)src_ptr[c + i];
                    params.dst_ptr[i] = (T*)dst_ptr[c + i];
                    FT_CHECK(size[c + i] % sizeof(T) == 0);
                    params.size[i] = size[c + i] / sizeof(T);
                }
                const int max_size = *std::max_element(params.size.begin(), params.size.end());
                dispatch(
                    std::integer_sequence<int, 1, 2, 4, 8, 16, 32, 64, 128>{},
                    [&](auto S) { return max_size <= S; },
                    BatchedCopyLauncher<BatchedCopyParam<T, C>>{max_size, count, &params, st});
            }
        });
}

template<int BLOCK_SIZE_>
__global__ void medusaBatchedMatch(const int* __restrict__ input_ids,
                                   const int* __restrict__ output_ids,
                                   const int* each_path_length,
                                   int*       match_length,
                                   int*       match_idx,
                                   const int  path_num,
                                   int        size)
{
    const int length  = size + 1;
    const int limit_r = gridDim.x * path_num * length;
    const int bid     = blockIdx.x;
    const int tid     = threadIdx.x;

    typedef cub::BlockReduce<TopK_2<int>, BLOCK_SIZE_> BlockReduce;
    __shared__ typename BlockReduce::TempStorage       temp_storage;

    TopK_2<int> partial;
    partial.init();

    for (int idx = tid; idx < path_num; idx += BLOCK_SIZE_) {
        int start_id = bid * path_num * length + idx * length;
        int limit_size_now;
        if (each_path_length) {
            limit_size_now = each_path_length[idx];
        }
        int accumulate_length = 0;
        for (int i = 0; i < limit_size_now && (start_id + i) < limit_r; ++i) {
            if (input_ids[start_id + i + 1] == output_ids[start_id + i]) {
                ++accumulate_length;
            }
            else {
                break;
            }
        }
        partial.insert(accumulate_length, idx);
    }

    TopK_2<int> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<int>);

    if (tid == 0) {
        const int index     = bid;
        match_length[index] = total.u;
        if (match_idx) {
            match_idx[index] = total.p;
        }
    }
    __syncthreads();
}

void invokeMedusaBatchMatch(const int*   input_ids,
                            const int*   output_ids,
                            const int*   each_path_length,
                            int*         max_match_length,
                            int*         max_match_idx,
                            int          batch_size,
                            int          path_num,
                            int          medusa_head_num,
                            cudaStream_t stream)
{
    dim3 grid, block;
    grid.x  = batch_size;
    block.x = 64;
    medusaBatchedMatch<64><<<grid, block, 0, stream>>>(
        input_ids, output_ids, each_path_length, max_match_length, max_match_idx, path_num, medusa_head_num);
}

template<typename T>
__global__ void compactCache(uintptr_t* cache_block_ptrs,
                             const int* sequence_length,
                             const int* medusa_verified_packed_idx,
                             const int* medusa_verified_length,
                             const int* cu_block_counts,
                             const int  kv_layer_num,
                             const int  kv_head_num,
                             const int  kv_cache_block_len,
                             const int  kv_head_dim,
                             const int  medusa_input_len,
                             const int  medusa_head_num,
                             const bool is_V)
{

    const int thread_idx  = threadIdx.x;
    const int batch_idx   = blockIdx.y / kv_layer_num;
    const int layer_idx   = blockIdx.y % kv_layer_num;
    const int kv_head_idx = blockIdx.z;

    const int match_count = medusa_verified_length[batch_idx];
    if (match_count == 0) {
        return;
    }

    uintptr_t* now_batch_block_ptrs     = cache_block_ptrs + cu_block_counts[batch_idx];
    int        layer_offset             = layer_idx * kv_head_num * kv_cache_block_len * 2 * kv_head_dim;
    int        head_offset              = kv_head_idx * kv_cache_block_len * kv_head_dim * 2;
    const int  first_generate_token_idx = sequence_length[batch_idx] - medusa_input_len;
    int        V_offset                 = is_V ? kv_cache_block_len * kv_head_dim : 0;

    for (int packed_i = 1; packed_i <= match_count; packed_i++) {
        int dst_packed_idx = *(medusa_verified_packed_idx + batch_idx * (1 + medusa_head_num) + packed_i);
        int dst_token_idx  = first_generate_token_idx + packed_i;        // dst
        int src_token_idx  = first_generate_token_idx + dst_packed_idx;  // src

        int dst_kv_block_idx       = dst_token_idx / kv_cache_block_len;
        int dst_token_kv_block_idx = dst_token_idx % kv_cache_block_len;
        int src_kv_block_idx       = src_token_idx / kv_cache_block_len;
        int src_token_kv_block_idx = src_token_idx % kv_cache_block_len;

        T* dst_block_ptr_start = (T*)(*(now_batch_block_ptrs + dst_kv_block_idx));
        T* src_block_ptr_start = (T*)(*(now_batch_block_ptrs + src_kv_block_idx));

        int src_token_offset = src_token_kv_block_idx * kv_head_dim;
        int dst_token_offset = dst_token_kv_block_idx * kv_head_dim;

        dst_block_ptr_start[layer_offset + head_offset + dst_token_offset + thread_idx + V_offset] =
            src_block_ptr_start[layer_offset + head_offset + src_token_offset + thread_idx + V_offset];
    }
}

template<typename T>
void invokeCompactKVCache(uintptr_t*   block_ptrs,
                          const int*   sequence_length,
                          const int*   medusa_verified_packed_idx,
                          const int*   medusa_verified_length,
                          const int*   cu_block_counts,
                          const int    kv_layer_num,
                          const int    kv_head_num,
                          const int    kv_cache_block_len,
                          const int    kv_head_dim,
                          const int    medusa_input_len,
                          const int    medusa_head_num,
                          const int    batch_size,
                          cudaStream_t stream)
{
    constexpr int block_size = 128;
    dim3          grid(1, batch_size * kv_layer_num, kv_head_num);
    // apply to K
    compactCache<T><<<grid, block_size, 0, stream>>>(block_ptrs,
                                                     sequence_length,
                                                     medusa_verified_packed_idx,
                                                     medusa_verified_length,
                                                     cu_block_counts,
                                                     kv_layer_num,
                                                     kv_head_num,
                                                     kv_cache_block_len,
                                                     kv_head_dim,
                                                     medusa_input_len,
                                                     medusa_head_num,
                                                     false);
    // apply to V
    compactCache<T><<<grid, block_size, 0, stream>>>(block_ptrs,
                                                     sequence_length,
                                                     medusa_verified_packed_idx,
                                                     medusa_verified_length,
                                                     cu_block_counts,
                                                     kv_layer_num,
                                                     kv_head_num,
                                                     kv_cache_block_len,
                                                     kv_head_dim,
                                                     medusa_input_len,
                                                     medusa_head_num,
                                                     true);
}
#define INSTANTIATE_INVOKE_COMPACT_KV_CACHE(T)                                                                         \
    template void invokeCompactKVCache<T>(uintptr_t * block_ptrs,                                                      \
                                          const int*   sequence_length,                                                \
                                          const int*   medusa_verified_packed_idx,                                     \
                                          const int*   medusa_verified_length,                                         \
                                          const int*   cu_block_counts,                                                \
                                          const int    kv_layer_num,                                                   \
                                          const int    kv_head_num,                                                    \
                                          const int    kv_cache_block_len,                                             \
                                          const int    kv_head_dim,                                                    \
                                          const int    medusa_input_len,                                               \
                                          const int    medusa_head_num,                                                \
                                          const int    batch_size,                                                     \
                                          cudaStream_t stream)
INSTANTIATE_INVOKE_COMPACT_KV_CACHE(half);
INSTANTIATE_INVOKE_COMPACT_KV_CACHE(float);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_COMPACT_KV_CACHE(__nv_bfloat16);
#endif
#undef INSTANTIATE_INVOKE_COMPACT_KV_CACHE

}  // namespace turbomind
