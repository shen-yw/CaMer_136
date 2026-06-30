# RVSD Token Merge Extension

Optional C++/CUDA extension for Co-Me-style grouped token contraction.

## Windows build

Open **Developer PowerShell for VS 2022**, activate the target conda env, then run:

```powershell
cd rvsd/ops/token_merge_ext
python setup.py build_ext --inplace -f
```

After a successful build, `rvsd.ops.token_merge_ext` will automatically use the compiled `ext` module at runtime.

If the extension is not built, RVSD automatically falls back to the eager PyTorch implementation.
