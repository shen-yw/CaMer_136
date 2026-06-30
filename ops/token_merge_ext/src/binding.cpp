#include <torch/extension.h>

#include "group_contract.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "RVSD token group contract extension";
    m.def(
        "group_contract_merge",
        &group_contract_merge,
        "Merge grouped tokens according to destination indices",
        pybind11::arg("grouped_tokens"),
        pybind11::arg("dst_index"),
        pybind11::arg("out_rows"));
    m.def(
        "group_contract_merge_unweighted_with_sizes",
        &group_contract_merge_unweighted_with_sizes,
        "Merge grouped tokens/coords and emit token sizes in one pass",
        pybind11::arg("grouped_tokens"),
        pybind11::arg("grouped_coords"),
        pybind11::arg("dst_index"),
        pybind11::arg("out_rows"),
        pybind11::arg("num_merge_groups"));
    m.def(
        "group_contract_merge_mask_unweighted_with_sizes",
        &group_contract_merge_mask_unweighted_with_sizes,
        "Fuse group index planning, token/coord contraction, old_to_new, and sizes",
        pybind11::arg("grouped_tokens"),
        pybind11::arg("grouped_coords"),
        pybind11::arg("merged_group_mask"),
        pybind11::arg("member_indices"),
        pybind11::arg("out_rows"),
        pybind11::arg("num_merge_groups"));
    m.def(
        "group_contract_scan",
        &group_contract_scan,
        "Build destination indices for grouped token contraction",
        pybind11::arg("merged_group_mask"),
        pybind11::arg("group_width"));
    m.def(
        "group_contract_plan",
        &group_contract_plan,
        "Build contraction dst_index and old_to_new in one pass",
        pybind11::arg("merged_group_mask"),
        pybind11::arg("member_indices"),
        pybind11::arg("group_width"),
        pybind11::arg("num_merge_groups"));
    m.def(
        "group_restore",
        &group_restore,
        "Restore merged patch tokens to original patch order",
        pybind11::arg("patch_tokens"),
        pybind11::arg("old_to_new"));
    m.def(
        "group_restore_to_map",
        &group_restore_to_map,
        "Restore merged patch tokens directly into a dense feature map",
        pybind11::arg("patch_tokens"),
        pybind11::arg("old_to_new"),
        pybind11::arg("patch_height"),
        pybind11::arg("patch_width"));
}
