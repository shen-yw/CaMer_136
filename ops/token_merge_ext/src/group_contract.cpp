#include "group_contract.h"

#include <vector>

namespace {

std::vector<torch::Tensor> group_contract_plan_cpu(
    torch::Tensor merged_group_mask,
    torch::Tensor member_indices,
    int64_t group_width,
    int64_t num_merge_groups);

torch::Tensor group_contract_merge_cpu(torch::Tensor grouped_tokens, torch::Tensor dst_index, int64_t out_rows) {
    TORCH_CHECK(grouped_tokens.dim() == 4, "grouped_tokens must be [B, N, G, C]");
    TORCH_CHECK(dst_index.dim() == 2, "dst_index must be [B, N*G]");
    TORCH_CHECK(grouped_tokens.size(0) == dst_index.size(0), "Batch size mismatch");

    const auto B = grouped_tokens.size(0);
    const auto N = grouped_tokens.size(1);
    const auto G = grouped_tokens.size(2);
    const auto C = grouped_tokens.size(3);
    TORCH_CHECK(dst_index.size(1) == N * G, "dst_index second dimension must equal N*G");

    auto flat_tokens = grouped_tokens.reshape({B, N * G, C});
    auto out = torch::zeros({B, out_rows, C}, grouped_tokens.options());
    auto scatter_index = dst_index.unsqueeze(-1).expand({B, N * G, C});
    out.scatter_add_(1, scatter_index, flat_tokens);

    auto counts = torch::zeros({B, out_rows, 1}, grouped_tokens.options());
    auto ones = torch::ones({B, N * G, 1}, grouped_tokens.options());
    counts.scatter_add_(1, dst_index.unsqueeze(-1), ones);
    return out / counts.clamp_min(1.0);
}

std::vector<torch::Tensor> group_contract_merge_unweighted_with_sizes_cpu(
    torch::Tensor grouped_tokens,
    torch::Tensor grouped_coords,
    torch::Tensor dst_index,
    int64_t out_rows,
    int64_t num_merge_groups) {
    TORCH_CHECK(grouped_tokens.dim() == 4, "grouped_tokens must be [B, N, G, C]");
    TORCH_CHECK(grouped_coords.dim() == 4, "grouped_coords must be [B, N, G, D]");
    TORCH_CHECK(grouped_tokens.size(0) == grouped_coords.size(0), "Batch size mismatch");
    TORCH_CHECK(grouped_tokens.size(1) == grouped_coords.size(1), "Group count mismatch");
    TORCH_CHECK(grouped_tokens.size(2) == grouped_coords.size(2), "Group width mismatch");
    auto merged_tokens = group_contract_merge_cpu(grouped_tokens, dst_index, out_rows);
    auto merged_coords = group_contract_merge_cpu(grouped_coords, dst_index, out_rows);
    auto token_sizes = torch::ones(
        {grouped_tokens.size(0), out_rows},
        grouped_tokens.options());
    if (num_merge_groups > 0) {
        token_sizes.narrow(1, 0, num_merge_groups).fill_(static_cast<double>(grouped_tokens.size(2)));
    }
    return {merged_tokens, merged_coords, token_sizes};
}

std::vector<torch::Tensor> group_contract_merge_mask_unweighted_with_sizes_cpu(
    torch::Tensor grouped_tokens,
    torch::Tensor grouped_coords,
    torch::Tensor merged_group_mask,
    torch::Tensor member_indices,
    int64_t out_rows,
    int64_t num_merge_groups) {
    const auto group_width = grouped_tokens.size(2);
    auto plan = group_contract_plan_cpu(merged_group_mask, member_indices, group_width, num_merge_groups);
    auto merged = group_contract_merge_unweighted_with_sizes_cpu(
        grouped_tokens,
        grouped_coords,
        plan[0],
        out_rows,
        num_merge_groups);
    return {merged[0], merged[1], plan[1], merged[2]};
}

torch::Tensor group_contract_scan_impl(torch::Tensor merged_group_mask, int64_t group_width) {
    TORCH_CHECK(merged_group_mask.dim() == 2, "merged_group_mask must be [B, N]");
    TORCH_CHECK(group_width > 0, "group_width must be positive");

    auto mask = merged_group_mask.to(torch::kBool).contiguous();
    auto mask_long = mask.to(torch::kLong);
    auto keep_long = mask.logical_not().to(torch::kLong);
    auto merged_ranks = torch::cumsum(mask_long, /*dim=*/1) - 1;
    auto kept_ranks = torch::cumsum(keep_long, /*dim=*/1) - 1;
    auto num_merge_groups = mask_long.sum(/*dim=*/1, /*keepdim=*/true);
    auto offsets = torch::arange(group_width, mask.options().dtype(torch::kLong)).view({1, 1, group_width});
    auto merged_index = merged_ranks.unsqueeze(-1);
    auto kept_index = num_merge_groups.unsqueeze(-1) + kept_ranks.unsqueeze(-1) * group_width + offsets;
    return torch::where(mask.unsqueeze(-1), merged_index, kept_index).reshape({mask.size(0), mask.size(1) * group_width});
}

std::vector<torch::Tensor> group_contract_plan_cpu(
    torch::Tensor merged_group_mask,
    torch::Tensor member_indices,
    int64_t group_width,
    int64_t num_merge_groups) {
    TORCH_CHECK(merged_group_mask.dim() == 2, "merged_group_mask must be [B, N]");
    TORCH_CHECK(member_indices.dim() == 2, "member_indices must be [N, G]");
    TORCH_CHECK(group_width > 0, "group_width must be positive");
    (void)num_merge_groups;
    TORCH_CHECK(member_indices.size(1) == group_width, "member_indices second dimension must equal group_width");
    auto dst_index = group_contract_scan_impl(merged_group_mask, group_width);
    auto old_to_new = torch::empty_like(dst_index);
    auto scatter_indices = member_indices.reshape({1, member_indices.size(0) * group_width}).expand_as(dst_index);
    old_to_new.scatter_(1, scatter_indices, dst_index);
    return {dst_index, old_to_new};
}

torch::Tensor group_restore_cpu(torch::Tensor patch_tokens, torch::Tensor old_to_new) {
    TORCH_CHECK(patch_tokens.dim() == 3, "patch_tokens must be [B, M, C]");
    TORCH_CHECK(old_to_new.dim() == 2, "old_to_new must be [B, N]");
    TORCH_CHECK(patch_tokens.size(0) == old_to_new.size(0), "Batch size mismatch");
    TORCH_CHECK(old_to_new.scalar_type() == torch::kLong, "old_to_new must be int64");

    const auto batch_size = patch_tokens.size(0);
    const auto num_original_tokens = old_to_new.size(1);
    const auto channels = patch_tokens.size(2);
    auto gather_index = old_to_new.unsqueeze(-1).expand({batch_size, num_original_tokens, channels});
    return patch_tokens.gather(1, gather_index);
}

torch::Tensor group_restore_to_map_cpu(
    torch::Tensor patch_tokens,
    torch::Tensor old_to_new,
    int64_t patch_height,
    int64_t patch_width) {
    TORCH_CHECK(patch_height > 0, "patch_height must be positive");
    TORCH_CHECK(patch_width > 0, "patch_width must be positive");
    auto restored = group_restore_cpu(patch_tokens, old_to_new);
    TORCH_CHECK(
        restored.size(1) == patch_height * patch_width,
        "old_to_new second dimension must equal patch_height * patch_width");
    return restored.reshape({restored.size(0), patch_height, patch_width, restored.size(2)})
        .permute({0, 3, 1, 2})
        .contiguous();
}

}  // namespace

#ifdef WITH_CUDA
torch::Tensor group_contract_merge_cuda(torch::Tensor grouped_tokens, torch::Tensor dst_index, int64_t out_rows);
std::vector<torch::Tensor> group_contract_merge_unweighted_with_sizes_cuda(
    torch::Tensor grouped_tokens,
    torch::Tensor grouped_coords,
    torch::Tensor dst_index,
    int64_t out_rows,
    int64_t num_merge_groups);
std::vector<torch::Tensor> group_contract_merge_mask_unweighted_with_sizes_cuda(
    torch::Tensor grouped_tokens,
    torch::Tensor grouped_coords,
    torch::Tensor merged_group_mask,
    torch::Tensor member_indices,
    int64_t out_rows,
    int64_t num_merge_groups);
std::vector<torch::Tensor> group_contract_plan_cuda(
    torch::Tensor merged_group_mask,
    torch::Tensor member_indices,
    int64_t group_width,
    int64_t num_merge_groups);
torch::Tensor group_restore_cuda(torch::Tensor patch_tokens, torch::Tensor old_to_new);
torch::Tensor group_restore_to_map_cuda(
    torch::Tensor patch_tokens,
    torch::Tensor old_to_new,
    int64_t patch_height,
    int64_t patch_width);
#endif

torch::Tensor group_contract_merge(torch::Tensor grouped_tokens, torch::Tensor dst_index, int64_t out_rows) {
    TORCH_CHECK(grouped_tokens.device() == dst_index.device(), "grouped_tokens and dst_index must be on same device");
    TORCH_CHECK(dst_index.scalar_type() == torch::kLong, "dst_index must be int64");
    TORCH_CHECK(out_rows > 0, "out_rows must be positive");

    if (grouped_tokens.is_cuda()) {
#ifdef WITH_CUDA
        return group_contract_merge_cuda(grouped_tokens.contiguous(), dst_index.contiguous(), out_rows);
#else
        TORCH_CHECK(false, "group_contract_merge was built without CUDA support");
#endif
    }

    return group_contract_merge_cpu(grouped_tokens.contiguous(), dst_index.contiguous(), out_rows);
}

std::vector<torch::Tensor> group_contract_merge_unweighted_with_sizes(
    torch::Tensor grouped_tokens,
    torch::Tensor grouped_coords,
    torch::Tensor dst_index,
    int64_t out_rows,
    int64_t num_merge_groups) {
    TORCH_CHECK(grouped_tokens.device() == grouped_coords.device(), "grouped_tokens and grouped_coords must be on same device");
    TORCH_CHECK(grouped_tokens.device() == dst_index.device(), "grouped_tokens and dst_index must be on same device");
    TORCH_CHECK(dst_index.scalar_type() == torch::kLong, "dst_index must be int64");
    TORCH_CHECK(out_rows > 0, "out_rows must be positive");
    TORCH_CHECK(num_merge_groups >= 0, "num_merge_groups must be non-negative");

    if (grouped_tokens.is_cuda()) {
#ifdef WITH_CUDA
        return group_contract_merge_unweighted_with_sizes_cuda(
            grouped_tokens.contiguous(),
            grouped_coords.contiguous(),
            dst_index.contiguous(),
            out_rows,
            num_merge_groups);
#else
        TORCH_CHECK(false, "group_contract_merge_unweighted_with_sizes was built without CUDA support");
#endif
    }

    return group_contract_merge_unweighted_with_sizes_cpu(
        grouped_tokens.contiguous(),
        grouped_coords.contiguous(),
        dst_index.contiguous(),
        out_rows,
        num_merge_groups);
}

std::vector<torch::Tensor> group_contract_merge_mask_unweighted_with_sizes(
    torch::Tensor grouped_tokens,
    torch::Tensor grouped_coords,
    torch::Tensor merged_group_mask,
    torch::Tensor member_indices,
    int64_t out_rows,
    int64_t num_merge_groups) {
    TORCH_CHECK(grouped_tokens.device() == grouped_coords.device(), "grouped_tokens and grouped_coords must be on same device");
    TORCH_CHECK(grouped_tokens.device() == merged_group_mask.device(), "grouped_tokens and merged_group_mask must be on same device");
    TORCH_CHECK(grouped_tokens.device() == member_indices.device(), "grouped_tokens and member_indices must be on same device");
    TORCH_CHECK(member_indices.scalar_type() == torch::kLong, "member_indices must be int64");
    TORCH_CHECK(out_rows > 0, "out_rows must be positive");
    TORCH_CHECK(num_merge_groups >= 0, "num_merge_groups must be non-negative");

    if (grouped_tokens.is_cuda()) {
#ifdef WITH_CUDA
        return group_contract_merge_mask_unweighted_with_sizes_cuda(
            grouped_tokens.contiguous(),
            grouped_coords.contiguous(),
            merged_group_mask.contiguous(),
            member_indices.contiguous(),
            out_rows,
            num_merge_groups);
#else
        TORCH_CHECK(false, "group_contract_merge_mask_unweighted_with_sizes was built without CUDA support");
#endif
    }

    return group_contract_merge_mask_unweighted_with_sizes_cpu(
        grouped_tokens.contiguous(),
        grouped_coords.contiguous(),
        merged_group_mask.contiguous(),
        member_indices.contiguous(),
        out_rows,
        num_merge_groups);
}

torch::Tensor group_contract_scan(torch::Tensor merged_group_mask, int64_t group_width) {
    return group_contract_scan_impl(merged_group_mask, group_width);
}

std::vector<torch::Tensor> group_contract_plan(
    torch::Tensor merged_group_mask,
    torch::Tensor member_indices,
    int64_t group_width,
    int64_t num_merge_groups) {
    TORCH_CHECK(merged_group_mask.device() == member_indices.device(), "merged_group_mask and member_indices must be on same device");
    TORCH_CHECK(member_indices.scalar_type() == torch::kLong, "member_indices must be int64");
    TORCH_CHECK(group_width > 0, "group_width must be positive");
    TORCH_CHECK(num_merge_groups >= 0, "num_merge_groups must be non-negative");

    if (merged_group_mask.is_cuda()) {
#ifdef WITH_CUDA
        return group_contract_plan_cuda(
            merged_group_mask.contiguous(),
            member_indices.contiguous(),
            group_width,
            num_merge_groups);
#else
        TORCH_CHECK(false, "group_contract_plan was built without CUDA support");
#endif
    }

    return group_contract_plan_cpu(
        merged_group_mask.contiguous(),
        member_indices.contiguous(),
        group_width,
        num_merge_groups);
}

torch::Tensor group_restore(torch::Tensor patch_tokens, torch::Tensor old_to_new) {
    TORCH_CHECK(patch_tokens.device() == old_to_new.device(), "patch_tokens and old_to_new must be on same device");
    TORCH_CHECK(old_to_new.scalar_type() == torch::kLong, "old_to_new must be int64");

    if (patch_tokens.is_cuda()) {
#ifdef WITH_CUDA
        return group_restore_cuda(patch_tokens.contiguous(), old_to_new.contiguous());
#else
        TORCH_CHECK(false, "group_restore was built without CUDA support");
#endif
    }

    return group_restore_cpu(patch_tokens.contiguous(), old_to_new.contiguous());
}

torch::Tensor group_restore_to_map(
    torch::Tensor patch_tokens,
    torch::Tensor old_to_new,
    int64_t patch_height,
    int64_t patch_width) {
    TORCH_CHECK(patch_tokens.device() == old_to_new.device(), "patch_tokens and old_to_new must be on same device");
    TORCH_CHECK(old_to_new.scalar_type() == torch::kLong, "old_to_new must be int64");
    TORCH_CHECK(patch_height > 0, "patch_height must be positive");
    TORCH_CHECK(patch_width > 0, "patch_width must be positive");

    if (patch_tokens.is_cuda()) {
#ifdef WITH_CUDA
        return group_restore_to_map_cuda(
            patch_tokens.contiguous(),
            old_to_new.contiguous(),
            patch_height,
            patch_width);
#else
        TORCH_CHECK(false, "group_restore_to_map was built without CUDA support");
#endif
    }

    return group_restore_to_map_cpu(
        patch_tokens.contiguous(),
        old_to_new.contiguous(),
        patch_height,
        patch_width);
}
