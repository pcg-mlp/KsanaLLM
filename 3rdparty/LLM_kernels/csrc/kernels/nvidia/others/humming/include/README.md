# Humming `.cuh` Headers (Local Copy)

These headers are a **verbatim copy** of the `include/humming/` directory from the
`humming-kernels` pip package, used for secondary development of the humming MoE
operator within KsanaLLM.

## Source

| Field | Value |
|-------|-------|
| Package | `humming-kernels` |
| Version | `0.1.5` |
| Commit | `d596b9f` |
| Source repo | https://github.com/inclusionAI/humming |
| Install | `pip install git+https://github.com/inclusionAI/humming.git@d596b9f` |
| Copied | 2026-06-18 |

## Why local?

To enable in-tree modification of the kernel implementations (memory pipeline,
WGMMA, stream-K scheduler, epilogue, etc.) for performance optimization. The
JIT compiler (`../jit/humming_jit_compiler.cpp`) and `../preprocess/CMakeLists.txt`
resolve includes from this directory.

## How to modify

1. Edit any `.cuh` file below.
2. Clear the JIT cache: `rm -rf ~/.cache/KsanaLLM/humming_kernel_files/`
3. Re-run the service / test. The next JIT compilation picks up your changes.

## How to upgrade

When upgrading `humming-kernels`, re-sync this directory from the new pip package:

```bash
HUMMING_PIP_INCLUDE=$(python3 -c "import humming; import os; print(os.path.join(os.path.dirname(humming.__file__), 'include'))")
rsync -a --delete "$HUMMING_PIP_INCLUDE/humming/" \
  3rdparty/LLM_kernels/csrc/kernels/nvidia/others/humming/include/humming/
```

Then review the diff and re-apply any local modifications. See
`../MIGRATION_TRACKING.md` for the full upgrade procedure.
