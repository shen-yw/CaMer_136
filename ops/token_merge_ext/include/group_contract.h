#pragma once

#include <torch/extension.h>
#include <vector>

torch::Tensor group_contract_merge(torch::Tensor grouped_tokens, torch::Tensor dst_index, int64_t out_rows);
std::vector<torch::Tensor> group_contract_merge_unweighted_with_sizes(
    torch::Tensor grouped_tokens,
    torch::Tensor grouped_coords,
    torch::Tensor dst_index,
    int64_t out_rows,
    int64_t num_merge_groups);
std::vector<torch::Tensor> group_contract_merge_mask_unweighted_with_sizes(
    torch::Tensor grouped_tokens,
    torch::Tensor grouped_coords,
    torch::Tensor merged_group_mask,
    torch::Tensor member_indices,
    int64_t out_rows,
    int64_t num_merge_groups);
torch::Tensor group_contract_scan(torch::Tensor merged_group_mask, int64_t group_width);
std::vector<torch::Tensor> group_contract_plan(
    torch::Tensor merged_group_mask,
    torch::Tensor member_indices,
    int64_t group_width,
    int64_t num_merge_groups);
torch::Tensor group_restore(torch::Tensor patch_tokens, torch::Tensor old_to_new);
torch::Tensor group_restore_to_map(
    torch::Tensor patch_tokens,
    torch::Tensor old_to_new,
    int64_t patch_height,
    int64_t patch_width);
