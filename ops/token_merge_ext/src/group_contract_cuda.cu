#include "group_contract.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

template <typename scalar_t>
__global__ void group_contract_kernel(
    const at::PackedTensorAccessor64<scalar_t, 4, at::RestrictPtrTraits> grouped_tokens,
    const at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> dst_index,
    at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> out,
    const int group_width,
    const int channels) {
    const int batch_id = blockIdx.x;
    const int group_id = blockIdx.y;
    const int row0 = group_id * group_width;

    __shared__ int64_t s_dst0;
    __shared__ int s_copy_case;
    if (threadIdx.x == 0) {
        const int64_t dst0 = dst_index[batch_id][row0];
        const int64_t dst1 = (row0 + 1 < dst_index.size(1)) ? dst_index[batch_id][row0 + 1] : (dst0 + group_width);
        s_dst0 = dst0;
        s_copy_case = (dst1 - dst0 == 1) ? 1 : 0;
    }
    __syncthreads();

    if (s_copy_case) {
        for (int c = threadIdx.x; c < channels; c += blockDim.x) {
            #pragma unroll
            for (int j = 0; j < group_width; ++j) {
                out[batch_id][s_dst0 + j][c] = grouped_tokens[batch_id][group_id][j][c];
            }
        }
    } else {
        for (int c = threadIdx.x; c < channels; c += blockDim.x) {
            scalar_t sum = scalar_t(0);
            #pragma unroll
            for (int j = 0; j < group_width; ++j) {
                sum += grouped_tokens[batch_id][group_id][j][c];
            }
            out[batch_id][s_dst0][c] = sum / static_cast<scalar_t>(group_width);
        }
    }
}

template <typename token_t, typename coord_t>
__global__ void group_contract_unweighted_with_sizes_kernel(
    const at::PackedTensorAccessor64<token_t, 4, at::RestrictPtrTraits> grouped_tokens,
    const at::PackedTensorAccessor64<coord_t, 4, at::RestrictPtrTraits> grouped_coords,
    const at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> dst_index,
    at::PackedTensorAccessor64<token_t, 3, at::RestrictPtrTraits> out_tokens,
    at::PackedTensorAccessor64<coord_t, 3, at::RestrictPtrTraits> out_coords,
    token_t* __restrict__ token_sizes,
    const int out_rows,
    const int num_merge_groups,
    const int group_width,
    const int channels,
    const int coord_channels) {
    const int batch_id = blockIdx.x;
    const int group_id = blockIdx.y;
    const int row0 = group_id * group_width;

    __shared__ int64_t s_dst0;
    __shared__ int s_copy_case;
    if (threadIdx.x == 0) {
        const int64_t dst0 = dst_index[batch_id][row0];
        const int64_t dst1 = (row0 + 1 < dst_index.size(1)) ? dst_index[batch_id][row0 + 1] : (dst0 + group_width);
        s_dst0 = dst0;
        s_copy_case = (dst1 - dst0 == 1) ? 1 : 0;
    }
    __syncthreads();

    if (s_copy_case) {
        for (int c = threadIdx.x; c < channels; c += blockDim.x) {
            #pragma unroll
            for (int j = 0; j < group_width; ++j) {
                out_tokens[batch_id][s_dst0 + j][c] = grouped_tokens[batch_id][group_id][j][c];
            }
        }
        for (int c = threadIdx.x; c < coord_channels; c += blockDim.x) {
            #pragma unroll
            for (int j = 0; j < group_width; ++j) {
                out_coords[batch_id][s_dst0 + j][c] = grouped_coords[batch_id][group_id][j][c];
            }
        }
        if (threadIdx.x == 0) {
            token_t* batch_sizes = token_sizes + batch_id * out_rows;
            #pragma unroll
            for (int j = 0; j < group_width; ++j) {
                batch_sizes[s_dst0 + j] = token_t(1);
            }
        }
    } else {
        for (int c = threadIdx.x; c < channels; c += blockDim.x) {
            token_t sum = token_t(0);
            #pragma unroll
            for (int j = 0; j < group_width; ++j) {
                sum += grouped_tokens[batch_id][group_id][j][c];
            }
            out_tokens[batch_id][s_dst0][c] = sum / static_cast<token_t>(group_width);
        }
        for (int c = threadIdx.x; c < coord_channels; c += blockDim.x) {
            coord_t sum = coord_t(0);
            #pragma unroll
            for (int j = 0; j < group_width; ++j) {
                sum += grouped_coords[batch_id][group_id][j][c];
            }
            out_coords[batch_id][s_dst0][c] = sum / static_cast<coord_t>(group_width);
        }
        if (threadIdx.x == 0) {
            token_sizes[batch_id * out_rows + s_dst0] = static_cast<token_t>(group_width);
        }
    }
}

template <typename token_t, typename coord_t>
__global__ void group_contract_mask_unweighted_with_sizes_kernel(
    const at::PackedTensorAccessor64<token_t, 4, at::RestrictPtrTraits> grouped_tokens,
    const at::PackedTensorAccessor64<coord_t, 4, at::RestrictPtrTraits> grouped_coords,
    const bool* __restrict__ merged_group_mask,
    const int64_t* __restrict__ member_indices,
    at::PackedTensorAccessor64<token_t, 3, at::RestrictPtrTraits> out_tokens,
    at::PackedTensorAccessor64<coord_t, 3, at::RestrictPtrTraits> out_coords,
    int64_t* __restrict__ old_to_new,
    token_t* __restrict__ token_sizes,
    const int out_rows,
    const int num_merge_groups,
    const int group_width,
    const int channels,
    const int coord_channels) {
    const int batch_id = blockIdx.x;
    const int group_id = blockIdx.y;
    const int num_groups = grouped_tokens.size(1);
    const bool* batch_mask = merged_group_mask + batch_id * num_groups;
    const bool should_merge = batch_mask[group_id];

    int merged_before = 0;
    for (int prev_group = 0; prev_group < group_id; ++prev_group) {
        merged_before += static_cast<int>(batch_mask[prev_group]);
    }
    const int kept_before = group_id - merged_before;
    const int dst0 = should_merge
        ? merged_before
        : num_merge_groups + kept_before * group_width;

    if (should_merge) {
        for (int c = threadIdx.x; c < channels; c += blockDim.x) {
            token_t sum = token_t(0);
            #pragma unroll
            for (int j = 0; j < group_width; ++j) {
                sum += grouped_tokens[batch_id][group_id][j][c];
            }
            out_tokens[batch_id][dst0][c] = sum / static_cast<token_t>(group_width);
        }
        for (int c = threadIdx.x; c < coord_channels; c += blockDim.x) {
            coord_t sum = coord_t(0);
            #pragma unroll
            for (int j = 0; j < group_width; ++j) {
                sum += grouped_coords[batch_id][group_id][j][c];
            }
            out_coords[batch_id][dst0][c] = sum / static_cast<coord_t>(group_width);
        }
        if (threadIdx.x == 0) {
            token_sizes[batch_id * out_rows + dst0] = static_cast<token_t>(group_width);
            const int flat_base = group_id * group_width;
            for (int j = 0; j < group_width; ++j) {
                const int64_t original_index = member_indices[flat_base + j];
                if (original_index >= 0) {
                    old_to_new[batch_id * num_groups * group_width + original_index] = dst0;
                }
            }
        }
    } else {
        for (int c = threadIdx.x; c < channels; c += blockDim.x) {
            #pragma unroll
            for (int j = 0; j < group_width; ++j) {
                out_tokens[batch_id][dst0 + j][c] = grouped_tokens[batch_id][group_id][j][c];
            }
        }
        for (int c = threadIdx.x; c < coord_channels; c += blockDim.x) {
            #pragma unroll
            for (int j = 0; j < group_width; ++j) {
                out_coords[batch_id][dst0 + j][c] = grouped_coords[batch_id][group_id][j][c];
            }
        }
        if (threadIdx.x == 0) {
            token_t* batch_sizes = token_sizes + batch_id * out_rows;
            const int flat_base = group_id * group_width;
            for (int j = 0; j < group_width; ++j) {
                batch_sizes[dst0 + j] = token_t(1);
                const int64_t original_index = member_indices[flat_base + j];
                if (original_index >= 0) {
                    old_to_new[batch_id * num_groups * group_width + original_index] = dst0 + j;
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void group_restore_kernel(
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> patch_tokens,
    const at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> old_to_new,
    at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> restored_tokens,
    const int channels) {
    const int batch_id = blockIdx.x;
    const int token_id = blockIdx.y;
    const int64_t source_id = old_to_new[batch_id][token_id];
    for (int c = threadIdx.x; c < channels; c += blockDim.x) {
        restored_tokens[batch_id][token_id][c] = patch_tokens[batch_id][source_id][c];
    }
}

__global__ void group_contract_plan_kernel(
    const bool* __restrict__ merged_group_mask,
    const int64_t* __restrict__ member_indices,
    int64_t* __restrict__ dst_index,
    int64_t* __restrict__ old_to_new,
    const int batch_size,
    const int num_groups,
    const int group_width,
    const int num_merge_groups) {
    const int linear_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * num_groups * group_width;
    if (linear_index >= total) {
        return;
    }

    const int member_offset = linear_index % group_width;
    const int group_id = (linear_index / group_width) % num_groups;
    const int batch_id = linear_index / (num_groups * group_width);
    const bool should_merge = merged_group_mask[batch_id * num_groups + group_id];

    int merged_before = 0;
    const bool* batch_mask = merged_group_mask + batch_id * num_groups;
    for (int prev_group = 0; prev_group < group_id; ++prev_group) {
        merged_before += static_cast<int>(batch_mask[prev_group]);
    }
    const int kept_before = group_id - merged_before;
    const int64_t dst = should_merge
        ? static_cast<int64_t>(merged_before)
        : static_cast<int64_t>(num_merge_groups + kept_before * group_width + member_offset);

    const int flat_member = group_id * group_width + member_offset;
    dst_index[batch_id * num_groups * group_width + flat_member] = dst;
    const int64_t original_index = member_indices[flat_member];
    if (original_index >= 0) {
        old_to_new[batch_id * num_groups * group_width + original_index] = dst;
    }
}

__global__ void group_contract_plan_prefix_kernel(
    const bool* __restrict__ merged_group_mask,
    const int64_t* __restrict__ member_indices,
    int64_t* __restrict__ dst_index,
    int64_t* __restrict__ old_to_new,
    const int num_groups,
    const int group_width,
    const int num_merge_groups) {
    extern __shared__ int merged_prefix[];
    const int batch_id = blockIdx.x;
    const int group_id = threadIdx.x;
    const bool* batch_mask = merged_group_mask + batch_id * num_groups;

    if (group_id < num_groups) {
        merged_prefix[group_id] = static_cast<int>(batch_mask[group_id]);
    }
    __syncthreads();

    for (int offset = 1; offset < num_groups; offset <<= 1) {
        int value = 0;
        if (group_id < num_groups && group_id >= offset) {
            value = merged_prefix[group_id - offset];
        }
        __syncthreads();
        if (group_id < num_groups) {
            merged_prefix[group_id] += value;
        }
        __syncthreads();
    }

    if (group_id >= num_groups) {
        return;
    }

    const bool should_merge = batch_mask[group_id];
    const int merged_before = merged_prefix[group_id] - static_cast<int>(should_merge);
    const int kept_before = group_id - merged_before;
    const int flat_base = group_id * group_width;
    for (int member_offset = 0; member_offset < group_width; ++member_offset) {
        const int64_t dst = should_merge
            ? static_cast<int64_t>(merged_before)
            : static_cast<int64_t>(num_merge_groups + kept_before * group_width + member_offset);
        const int flat_member = flat_base + member_offset;
        dst_index[batch_id * num_groups * group_width + flat_member] = dst;
        const int64_t original_index = member_indices[flat_member];
        if (original_index >= 0) {
            old_to_new[batch_id * num_groups * group_width + original_index] = dst;
        }
    }
}

template <typename scalar_t>
__global__ void group_restore_to_map_kernel(
    const scalar_t* __restrict__ patch_tokens,
    const int64_t* __restrict__ old_to_new,
    scalar_t* __restrict__ feature_map,
    const int batch_size,
    const int sparse_tokens,
    const int num_original_tokens,
    const int channels) {
    const int linear_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * channels * num_original_tokens;
    if (linear_index >= total) {
        return;
    }

    const int token_id = linear_index % num_original_tokens;
    const int channel_id = (linear_index / num_original_tokens) % channels;
    const int batch_id = linear_index / (num_original_tokens * channels);
    const int64_t source_id = old_to_new[batch_id * num_original_tokens + token_id];
    feature_map[linear_index] = patch_tokens[(batch_id * sparse_tokens + source_id) * channels + channel_id];
}

torch::Tensor group_contract_merge_cuda(torch::Tensor grouped_tokens, torch::Tensor dst_index, int64_t out_rows) {
    TORCH_CHECK(grouped_tokens.is_cuda(), "grouped_tokens must be CUDA");
    TORCH_CHECK(dst_index.is_cuda(), "dst_index must be CUDA");
    TORCH_CHECK(grouped_tokens.dim() == 4, "grouped_tokens must be [B, N, G, C]");
    TORCH_CHECK(dst_index.dim() == 2, "dst_index must be [B, N*G]");

    const int64_t batch_size = grouped_tokens.size(0);
    const int64_t num_groups = grouped_tokens.size(1);
    const int64_t group_width = grouped_tokens.size(2);
    const int64_t channels = grouped_tokens.size(3);
    TORCH_CHECK(dst_index.size(0) == batch_size, "Batch size mismatch");
    TORCH_CHECK(dst_index.size(1) == num_groups * group_width, "dst_index second dimension must equal N*G");

    auto out = torch::empty({batch_size, out_rows, channels}, grouped_tokens.options());
    constexpr int threads = 256;
    const dim3 block(threads);
    const dim3 grid(batch_size, num_groups);
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto scalar_type = grouped_tokens.scalar_type();
    if (scalar_type == at::kFloat) {
        group_contract_kernel<float><<<grid, block, 0, stream>>>(
            grouped_tokens.packed_accessor64<float, 4, at::RestrictPtrTraits>(),
            dst_index.packed_accessor64<int64_t, 2, at::RestrictPtrTraits>(),
            out.packed_accessor64<float, 3, at::RestrictPtrTraits>(),
            static_cast<int>(group_width),
            static_cast<int>(channels));
    } else if (scalar_type == at::kHalf) {
        group_contract_kernel<at::Half><<<grid, block, 0, stream>>>(
            grouped_tokens.packed_accessor64<at::Half, 4, at::RestrictPtrTraits>(),
            dst_index.packed_accessor64<int64_t, 2, at::RestrictPtrTraits>(),
            out.packed_accessor64<at::Half, 3, at::RestrictPtrTraits>(),
            static_cast<int>(group_width),
            static_cast<int>(channels));
    } else {
        TORCH_CHECK(
            false,
            "group_contract_merge_cuda supports float32/float16 tensors, got ",
            grouped_tokens.scalar_type()
        );
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "group_contract_kernel launch failed: ", cudaGetErrorString(err));
    return out;
}

std::vector<torch::Tensor> group_contract_merge_unweighted_with_sizes_cuda(
    torch::Tensor grouped_tokens,
    torch::Tensor grouped_coords,
    torch::Tensor dst_index,
    int64_t out_rows,
    int64_t num_merge_groups) {
    TORCH_CHECK(grouped_tokens.is_cuda(), "grouped_tokens must be CUDA");
    TORCH_CHECK(grouped_coords.is_cuda(), "grouped_coords must be CUDA");
    TORCH_CHECK(dst_index.is_cuda(), "dst_index must be CUDA");
    TORCH_CHECK(grouped_tokens.dim() == 4, "grouped_tokens must be [B, N, G, C]");
    TORCH_CHECK(grouped_coords.dim() == 4, "grouped_coords must be [B, N, G, D]");
    TORCH_CHECK(dst_index.dim() == 2, "dst_index must be [B, N*G]");

    const int64_t batch_size = grouped_tokens.size(0);
    const int64_t num_groups = grouped_tokens.size(1);
    const int64_t group_width = grouped_tokens.size(2);
    const int64_t channels = grouped_tokens.size(3);
    const int64_t coord_channels = grouped_coords.size(3);
    TORCH_CHECK(grouped_coords.size(0) == batch_size, "Batch size mismatch");
    TORCH_CHECK(grouped_coords.size(1) == num_groups, "Group count mismatch");
    TORCH_CHECK(grouped_coords.size(2) == group_width, "Group width mismatch");
    TORCH_CHECK(dst_index.size(0) == batch_size, "Batch size mismatch");
    TORCH_CHECK(dst_index.size(1) == num_groups * group_width, "dst_index second dimension must equal N*G");
    TORCH_CHECK(out_rows > 0, "out_rows must be positive");
    TORCH_CHECK(num_merge_groups >= 0 && num_merge_groups <= out_rows, "num_merge_groups is out of range");

    auto out_tokens = torch::empty({batch_size, out_rows, channels}, grouped_tokens.options());
    auto out_coords = torch::empty({batch_size, out_rows, coord_channels}, grouped_coords.options());
    auto token_sizes = torch::empty({batch_size, out_rows}, grouped_tokens.options());
    constexpr int threads = 256;
    const dim3 block(threads);
    const dim3 grid(batch_size, num_groups);
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto token_type = grouped_tokens.scalar_type();
    const auto coord_type = grouped_coords.scalar_type();
#define DISPATCH_COORD_TYPE(TOKEN_CPP_TYPE)                                                                           \
    if (coord_type == at::kFloat) {                                                                                   \
        group_contract_unweighted_with_sizes_kernel<TOKEN_CPP_TYPE, float><<<grid, block, 0, stream>>>(               \
            grouped_tokens.packed_accessor64<TOKEN_CPP_TYPE, 4, at::RestrictPtrTraits>(),                             \
            grouped_coords.packed_accessor64<float, 4, at::RestrictPtrTraits>(),                                      \
            dst_index.packed_accessor64<int64_t, 2, at::RestrictPtrTraits>(),                                         \
            out_tokens.packed_accessor64<TOKEN_CPP_TYPE, 3, at::RestrictPtrTraits>(),                                 \
            out_coords.packed_accessor64<float, 3, at::RestrictPtrTraits>(),                                          \
            token_sizes.data_ptr<TOKEN_CPP_TYPE>(),                                                                   \
            static_cast<int>(out_rows),                                                                               \
            static_cast<int>(num_merge_groups),                                                                       \
            static_cast<int>(group_width),                                                                            \
            static_cast<int>(channels),                                                                               \
            static_cast<int>(coord_channels));                                                                        \
    } else if (coord_type == at::kHalf) {                                                                             \
        group_contract_unweighted_with_sizes_kernel<TOKEN_CPP_TYPE, at::Half><<<grid, block, 0, stream>>>(            \
            grouped_tokens.packed_accessor64<TOKEN_CPP_TYPE, 4, at::RestrictPtrTraits>(),                             \
            grouped_coords.packed_accessor64<at::Half, 4, at::RestrictPtrTraits>(),                                   \
            dst_index.packed_accessor64<int64_t, 2, at::RestrictPtrTraits>(),                                         \
            out_tokens.packed_accessor64<TOKEN_CPP_TYPE, 3, at::RestrictPtrTraits>(),                                 \
            out_coords.packed_accessor64<at::Half, 3, at::RestrictPtrTraits>(),                                       \
            token_sizes.data_ptr<TOKEN_CPP_TYPE>(),                                                                   \
            static_cast<int>(out_rows),                                                                               \
            static_cast<int>(num_merge_groups),                                                                       \
            static_cast<int>(group_width),                                                                            \
            static_cast<int>(channels),                                                                               \
            static_cast<int>(coord_channels));                                                                        \
    } else {                                                                                                          \
        TORCH_CHECK(false, "grouped_coords must be float32/float16, got ", coord_type);                              \
    }

    if (token_type == at::kFloat) {
        DISPATCH_COORD_TYPE(float)
    } else if (token_type == at::kHalf) {
        DISPATCH_COORD_TYPE(at::Half)
    } else {
        TORCH_CHECK(false, "grouped_tokens must be float32/float16, got ", token_type);
    }
#undef DISPATCH_COORD_TYPE

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "group_contract_unweighted_with_sizes_kernel launch failed: ", cudaGetErrorString(err));
    return {out_tokens, out_coords, token_sizes};
}

std::vector<torch::Tensor> group_contract_merge_mask_unweighted_with_sizes_cuda(
    torch::Tensor grouped_tokens,
    torch::Tensor grouped_coords,
    torch::Tensor merged_group_mask,
    torch::Tensor member_indices,
    int64_t out_rows,
    int64_t num_merge_groups) {
    TORCH_CHECK(grouped_tokens.is_cuda(), "grouped_tokens must be CUDA");
    TORCH_CHECK(grouped_coords.is_cuda(), "grouped_coords must be CUDA");
    TORCH_CHECK(merged_group_mask.is_cuda(), "merged_group_mask must be CUDA");
    TORCH_CHECK(member_indices.is_cuda(), "member_indices must be CUDA");
    TORCH_CHECK(grouped_tokens.dim() == 4, "grouped_tokens must be [B, N, G, C]");
    TORCH_CHECK(grouped_coords.dim() == 4, "grouped_coords must be [B, N, G, D]");
    TORCH_CHECK(merged_group_mask.dim() == 2, "merged_group_mask must be [B, N]");
    TORCH_CHECK(member_indices.dim() == 2, "member_indices must be [N, G]");
    TORCH_CHECK(member_indices.scalar_type() == torch::kLong, "member_indices must be int64");
    TORCH_CHECK(merged_group_mask.scalar_type() == torch::kBool, "merged_group_mask must be bool");

    const int64_t batch_size = grouped_tokens.size(0);
    const int64_t num_groups = grouped_tokens.size(1);
    const int64_t group_width = grouped_tokens.size(2);
    const int64_t channels = grouped_tokens.size(3);
    const int64_t coord_channels = grouped_coords.size(3);
    TORCH_CHECK(grouped_coords.size(0) == batch_size, "Batch size mismatch");
    TORCH_CHECK(grouped_coords.size(1) == num_groups, "Group count mismatch");
    TORCH_CHECK(grouped_coords.size(2) == group_width, "Group width mismatch");
    TORCH_CHECK(merged_group_mask.size(0) == batch_size, "Mask batch size mismatch");
    TORCH_CHECK(merged_group_mask.size(1) == num_groups, "Mask group count mismatch");
    TORCH_CHECK(member_indices.size(0) == num_groups, "member_indices first dimension must equal num_groups");
    TORCH_CHECK(member_indices.size(1) == group_width, "member_indices second dimension must equal group_width");
    TORCH_CHECK(out_rows > 0, "out_rows must be positive");
    TORCH_CHECK(num_merge_groups >= 0 && num_merge_groups <= out_rows, "num_merge_groups is out of range");

    auto out_tokens = torch::empty({batch_size, out_rows, channels}, grouped_tokens.options());
    auto out_coords = torch::empty({batch_size, out_rows, coord_channels}, grouped_coords.options());
    auto old_to_new = torch::empty({batch_size, num_groups * group_width}, member_indices.options());
    auto token_sizes = torch::empty({batch_size, out_rows}, grouped_tokens.options());
    constexpr int threads = 256;
    const dim3 block(threads);
    const dim3 grid(batch_size, num_groups);
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto token_type = grouped_tokens.scalar_type();
    const auto coord_type = grouped_coords.scalar_type();
#define DISPATCH_MASK_COORD_TYPE(TOKEN_CPP_TYPE)                                                                      \
    if (coord_type == at::kFloat) {                                                                                   \
        group_contract_mask_unweighted_with_sizes_kernel<TOKEN_CPP_TYPE, float><<<grid, block, 0, stream>>>(          \
            grouped_tokens.packed_accessor64<TOKEN_CPP_TYPE, 4, at::RestrictPtrTraits>(),                             \
            grouped_coords.packed_accessor64<float, 4, at::RestrictPtrTraits>(),                                      \
            merged_group_mask.data_ptr<bool>(),                                                                       \
            member_indices.data_ptr<int64_t>(),                                                                       \
            out_tokens.packed_accessor64<TOKEN_CPP_TYPE, 3, at::RestrictPtrTraits>(),                                 \
            out_coords.packed_accessor64<float, 3, at::RestrictPtrTraits>(),                                          \
            old_to_new.data_ptr<int64_t>(),                                                                           \
            token_sizes.data_ptr<TOKEN_CPP_TYPE>(),                                                                   \
            static_cast<int>(out_rows),                                                                               \
            static_cast<int>(num_merge_groups),                                                                       \
            static_cast<int>(group_width),                                                                            \
            static_cast<int>(channels),                                                                               \
            static_cast<int>(coord_channels));                                                                        \
    } else if (coord_type == at::kHalf) {                                                                             \
        group_contract_mask_unweighted_with_sizes_kernel<TOKEN_CPP_TYPE, at::Half><<<grid, block, 0, stream>>>(       \
            grouped_tokens.packed_accessor64<TOKEN_CPP_TYPE, 4, at::RestrictPtrTraits>(),                             \
            grouped_coords.packed_accessor64<at::Half, 4, at::RestrictPtrTraits>(),                                   \
            merged_group_mask.data_ptr<bool>(),                                                                       \
            member_indices.data_ptr<int64_t>(),                                                                       \
            out_tokens.packed_accessor64<TOKEN_CPP_TYPE, 3, at::RestrictPtrTraits>(),                                 \
            out_coords.packed_accessor64<at::Half, 3, at::RestrictPtrTraits>(),                                       \
            old_to_new.data_ptr<int64_t>(),                                                                           \
            token_sizes.data_ptr<TOKEN_CPP_TYPE>(),                                                                   \
            static_cast<int>(out_rows),                                                                               \
            static_cast<int>(num_merge_groups),                                                                       \
            static_cast<int>(group_width),                                                                            \
            static_cast<int>(channels),                                                                               \
            static_cast<int>(coord_channels));                                                                        \
    } else {                                                                                                          \
        TORCH_CHECK(false, "grouped_coords must be float32/float16, got ", coord_type);                              \
    }

    if (token_type == at::kFloat) {
        DISPATCH_MASK_COORD_TYPE(float)
    } else if (token_type == at::kHalf) {
        DISPATCH_MASK_COORD_TYPE(at::Half)
    } else {
        TORCH_CHECK(false, "grouped_tokens must be float32/float16, got ", token_type);
    }
#undef DISPATCH_MASK_COORD_TYPE

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "group_contract_mask_unweighted_with_sizes_kernel launch failed: ", cudaGetErrorString(err));
    return {out_tokens, out_coords, old_to_new, token_sizes};
}

std::vector<torch::Tensor> group_contract_plan_cuda(
    torch::Tensor merged_group_mask,
    torch::Tensor member_indices,
    int64_t group_width,
    int64_t num_merge_groups) {
    TORCH_CHECK(merged_group_mask.is_cuda(), "merged_group_mask must be CUDA");
    TORCH_CHECK(member_indices.is_cuda(), "member_indices must be CUDA");
    TORCH_CHECK(merged_group_mask.dim() == 2, "merged_group_mask must be [B, N]");
    TORCH_CHECK(member_indices.dim() == 2, "member_indices must be [N, G]");
    TORCH_CHECK(member_indices.scalar_type() == torch::kLong, "member_indices must be int64");
    TORCH_CHECK(group_width > 0, "group_width must be positive");
    TORCH_CHECK(member_indices.size(1) == group_width, "member_indices second dimension must equal group_width");
    TORCH_CHECK(member_indices.size(0) == merged_group_mask.size(1), "member_indices first dimension must equal num_groups");
    TORCH_CHECK(num_merge_groups >= 0, "num_merge_groups must be non-negative");

    auto mask = merged_group_mask.to(torch::kBool).contiguous();
    const int64_t batch_size = mask.size(0);
    const int64_t num_groups = mask.size(1);
    const int64_t flat_tokens = num_groups * group_width;
    auto dst_index = torch::empty({batch_size, flat_tokens}, member_indices.options());
    auto old_to_new = torch::empty({batch_size, flat_tokens}, member_indices.options());
    auto stream = at::cuda::getCurrentCUDAStream();

    if (num_groups <= 1024) {
        int threads = 1;
        while (threads < num_groups) {
            threads <<= 1;
        }
        const int shared_bytes = static_cast<int>(num_groups * sizeof(int));
        group_contract_plan_prefix_kernel<<<static_cast<int>(batch_size), threads, shared_bytes, stream>>>(
            mask.data_ptr<bool>(),
            member_indices.data_ptr<int64_t>(),
            dst_index.data_ptr<int64_t>(),
            old_to_new.data_ptr<int64_t>(),
            static_cast<int>(num_groups),
            static_cast<int>(group_width),
            static_cast<int>(num_merge_groups));
    } else {
        constexpr int threads = 256;
        const int64_t total = batch_size * flat_tokens;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        group_contract_plan_kernel<<<blocks, threads, 0, stream>>>(
            mask.data_ptr<bool>(),
            member_indices.data_ptr<int64_t>(),
            dst_index.data_ptr<int64_t>(),
            old_to_new.data_ptr<int64_t>(),
            static_cast<int>(batch_size),
            static_cast<int>(num_groups),
            static_cast<int>(group_width),
            static_cast<int>(num_merge_groups));
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "group_contract_plan_kernel launch failed: ", cudaGetErrorString(err));
    return {dst_index, old_to_new};
}

torch::Tensor group_restore_cuda(torch::Tensor patch_tokens, torch::Tensor old_to_new) {
    TORCH_CHECK(patch_tokens.is_cuda(), "patch_tokens must be CUDA");
    TORCH_CHECK(old_to_new.is_cuda(), "old_to_new must be CUDA");
    TORCH_CHECK(patch_tokens.dim() == 3, "patch_tokens must be [B, M, C]");
    TORCH_CHECK(old_to_new.dim() == 2, "old_to_new must be [B, N]");
    TORCH_CHECK(old_to_new.scalar_type() == torch::kLong, "old_to_new must be int64");
    TORCH_CHECK(patch_tokens.size(0) == old_to_new.size(0), "Batch size mismatch");

    const int64_t batch_size = patch_tokens.size(0);
    const int64_t num_original_tokens = old_to_new.size(1);
    const int64_t channels = patch_tokens.size(2);

    auto out = torch::empty({batch_size, num_original_tokens, channels}, patch_tokens.options());
    constexpr int threads = 256;
    const dim3 block(threads);
    const dim3 grid(batch_size, num_original_tokens);
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto scalar_type = patch_tokens.scalar_type();
    if (scalar_type == at::kFloat) {
        group_restore_kernel<float><<<grid, block, 0, stream>>>(
            patch_tokens.packed_accessor64<float, 3, at::RestrictPtrTraits>(),
            old_to_new.packed_accessor64<int64_t, 2, at::RestrictPtrTraits>(),
            out.packed_accessor64<float, 3, at::RestrictPtrTraits>(),
            static_cast<int>(channels));
    } else if (scalar_type == at::kHalf) {
        group_restore_kernel<at::Half><<<grid, block, 0, stream>>>(
            patch_tokens.packed_accessor64<at::Half, 3, at::RestrictPtrTraits>(),
            old_to_new.packed_accessor64<int64_t, 2, at::RestrictPtrTraits>(),
            out.packed_accessor64<at::Half, 3, at::RestrictPtrTraits>(),
            static_cast<int>(channels));
    } else {
        TORCH_CHECK(false, "group_restore_cuda supports float32/float16 tensors, got ", scalar_type);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "group_restore_kernel launch failed: ", cudaGetErrorString(err));
    return out;
}

torch::Tensor group_restore_to_map_cuda(
    torch::Tensor patch_tokens,
    torch::Tensor old_to_new,
    int64_t patch_height,
    int64_t patch_width) {
    TORCH_CHECK(patch_tokens.is_cuda(), "patch_tokens must be CUDA");
    TORCH_CHECK(old_to_new.is_cuda(), "old_to_new must be CUDA");
    TORCH_CHECK(patch_tokens.dim() == 3, "patch_tokens must be [B, M, C]");
    TORCH_CHECK(old_to_new.dim() == 2, "old_to_new must be [B, N]");
    TORCH_CHECK(old_to_new.scalar_type() == torch::kLong, "old_to_new must be int64");
    TORCH_CHECK(patch_tokens.size(0) == old_to_new.size(0), "Batch size mismatch");
    TORCH_CHECK(patch_height > 0, "patch_height must be positive");
    TORCH_CHECK(patch_width > 0, "patch_width must be positive");

    const int64_t batch_size = patch_tokens.size(0);
    const int64_t sparse_tokens = patch_tokens.size(1);
    const int64_t num_original_tokens = old_to_new.size(1);
    const int64_t channels = patch_tokens.size(2);
    TORCH_CHECK(
        num_original_tokens == patch_height * patch_width,
        "old_to_new second dimension must equal patch_height * patch_width");

    auto out = torch::empty({batch_size, channels, patch_height, patch_width}, patch_tokens.options());
    constexpr int threads = 256;
    const int64_t total = batch_size * channels * num_original_tokens;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto scalar_type = patch_tokens.scalar_type();
    if (scalar_type == at::kFloat) {
        group_restore_to_map_kernel<float><<<blocks, threads, 0, stream>>>(
            patch_tokens.data_ptr<float>(),
            old_to_new.data_ptr<int64_t>(),
            out.data_ptr<float>(),
            static_cast<int>(batch_size),
            static_cast<int>(sparse_tokens),
            static_cast<int>(num_original_tokens),
            static_cast<int>(channels));
    } else if (scalar_type == at::kHalf) {
        group_restore_to_map_kernel<at::Half><<<blocks, threads, 0, stream>>>(
            patch_tokens.data_ptr<at::Half>(),
            old_to_new.data_ptr<int64_t>(),
            out.data_ptr<at::Half>(),
            static_cast<int>(batch_size),
            static_cast<int>(sparse_tokens),
            static_cast<int>(num_original_tokens),
            static_cast<int>(channels));
    } else {
        TORCH_CHECK(false, "group_restore_to_map_cuda supports float32/float16 tensors, got ", scalar_type);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "group_restore_to_map_kernel launch failed: ", cudaGetErrorString(err));
    return out;
}
