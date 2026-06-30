from __future__ import annotations

import os
from pathlib import Path
import sys

import torch
from setuptools import setup
import torch.utils.cpp_extension as cpp_extension
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on", "y"}


if _truthy_env("RVSD_TOKEN_MERGE_EXT_ALLOW_CUDA_MISMATCH"):
    cpp_extension._check_cuda_version = lambda *args, **kwargs: None
    print(
        "[token_merge_ext] RVSD_TOKEN_MERGE_EXT_ALLOW_CUDA_MISMATCH=1; "
        "skipping PyTorch CUDA toolkit version check."
    )


def get_extensions():
    this_dir = Path(__file__).resolve().parent
    src_dir = this_dir / "src"
    include_dir = this_dir / "include"

    cpp_sources = [str(src_dir / "binding.cpp"), str(src_dir / "group_contract.cpp")]
    cuda_source = str(src_dir / "group_contract_cuda.cu")

    use_cuda = torch.cuda.is_available() and CUDA_HOME is not None and Path(cuda_source).exists()
    extension_cls = CUDAExtension if use_cuda else CppExtension
    sources = cpp_sources + ([cuda_source] if use_cuda else [])
    define_macros = [("WITH_CUDA", None)] if use_cuda else []
    extra_compile_args = {"cxx": ["/O2"] if sys.platform == "win32" else ["-O3"]}
    if use_cuda:
        extra_compile_args["nvcc"] = ["-O3", "--use_fast_math"]

    return [
        extension_cls(
            name="ext",
            sources=sources,
            include_dirs=[str(include_dir)],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]


setup(
    name="rvsd-token-merge-ext",
    version="0.1.0",
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)
