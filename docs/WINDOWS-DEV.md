# Windows development environment map

C:\dev is the placeholder for any personal machine root. Machine-specific
paths never enter Git; replace them with your own when reproducing. When a
configuration fails, the authoritative source of truth is the *working*
build directory's `build/<preset>/CMakeCache.txt` — copy missing cache
entries from there line by line instead of guessing.

## HIP build (no preset)

Presets do not carry the HIP toolchain; plain `cmake` with every required
`-D` works. All HIP toolchain files come from an TheRock-style local LLVM/HIP
tree, below `<HIP_ROOT>\build` (e.g. `C:\dev\TheRock\build`):

```
cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBUILD_TESTING=OFF -DENGINE_ENABLE_HIP=ON -DGUFO_ENABLE_WARNINGS=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake \
  -DCMAKE_C_COMPILER='<HIP_ROOT>/build/lib/llvm/bin/clang.exe' \
  -DCMAKE_CXX_COMPILER='<HIP_ROOT>/build/lib/llvm/bin/clang++.exe' \
  -DCMAKE_HIP_COMPILER='<HIP_ROOT>/build/lib/llvm/bin/clang++.exe' \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_RC_COMPILER='<winkits>/Windows Kits/10/bin/<ver>/x64/rc.exe' \
  -DCMAKE_PREFIX_PATH='<HIP_ROOT>/build'
```

Facts that bite:

- C/C++/HIP must all be clang; mixing MSVC C++ with clang HIP is a configure
  error (`__verify_same_language_values`).
- `-D` together with `--preset` silently leaves the flag UNINITIALIZED
  (unapplied). Use presets or plain flags, not both.
- Dependencies come from vcpkg (ICU among them); the toolchain file is not
  optional. `$env:VCPKG_ROOT` alone does not substitute for it.
- The repo's `compat/win32/hip_include` tree reaches the HIP compiler via
  `CMAKE_HIP_FLAGS` (set by the repo CMake lists, not by the caller).
- If an executable dies instantly with `0xC0000135` (STATUS_DLL_NOT_FOUND),
  its build directory is missing runtime DLLs. CMake copies the vcpkg set
  (ICU, jpeg, png, ...); the HIP runtime set (`amdhip64_7.dll`,
  `hipblas.dll`, `libhipblaslt.dll`, `rocblas.dll`, `rocsolver.dll`,
  `rocsparse.dll`, `roc*-openblas*.dll`, `libssl-3-x64.dll`, ...) must be
  next to `gufo.exe`. Copying the full DLL set from any previously working
  build directory is the quick fix.

## NPU build and tests

- Opt in with `-DGUFO_ENABLE_NPU=ON` plus
  `-DGUFO_XRT_TOOLING_ROOT=<xrt-tooling-root>`; the root holds the pinned XRT
  headers/libs, and the two hash-pinned fake-XRT artifacts
  `final_4096x2560x1024_128x64x128.xclbin` +
  `insts_4096x2560x1024_128x64x128.txt` from `tools/npu/` must sit at the
  root itself (not in subdirectories) for `npu_private_bo_adapter_test`.
- NPU sources live with the model:
  `src\models\qwen38_flash_next\npu\`, wired via `cmake\Npu.cmake`.
- `tests\npu\` suites run in any preset with HIP (`hip::amdhip64`): the
  fake-XRT/fake-HIP binaries, 20 `npu_session_*` pytest wrappers and the
  pure-CPU fault targets. `libcrypto-3-x64.dll` next to `npu_*_test.exe` is
  derived in CMake from `OPENSSL_INCLUDE_DIR` automatically.
- Design and usage documentation: `docs\npu.md` plus the design notes inside
  the npu directory.

## Quick benchmark

Benchmarks use the production command line; no thinking switch exists.

```
gufo bench --model '<hf-cache>\models--unsloth--Qwen3.8-Flash-Next-GGUF\
  <snapshot>\UD-Q4_K_XL\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf'
  --n-prompt 2048 --n-gen 128 --n-depth 0,16384 --repetitions 1
  --temperature 0
```

- Model files live in the local HF cache for
  `unsloth/Qwen3.8-Flash-Next-GGUF`; use the newest snapshot that contains
  the quantization (4-part UD-Q4_K_XL plus an `MTP` sidecar and
  `mmproj-BF16.gguf`).
- `tools\windows\run-flash-next.ps1 -Mode bench` wraps the full sweep
  (depths `0,4096,16384,32768,65536,131072`, `--repetitions 2`, specular
  args). `-Draft off` = AR baseline, `-Greedy` = `--temperature 0`.
- Running two binaries against each other: run them sequentially, never
  concurrently, and unset `GUFO_NPU*` environment for AR comparisons.

## Runtime DLL roots (test and serve binaries)

Built executables resolve shared libraries at launch. Exit code
`0xC0000135` (DLL not found) means a root below is missing from `PATH`; the
build itself is fine. Find any binary's full list with
`dumpbin /DEPENDENTS <exe>` and check it against these roots:

- `C:\dev\TheRock\build\bin` — `amdhip64_7.dll`, `hipblas`, `rocblas`,
  `libhipblaslt`, `MIOpen`, `hiprtc0715` and friends.
- `C:\dev\vcpkg\installed\x64-windows\bin` — `libcurl-d`, `libcrypto-3-x64`,
  `libpng16`, `jpeg62`, `icuuc78`/`icuin78`.
- The XRT runtime providing `xrt_coreutil.dll` — linked by everything that
  pulls in NPU staging: `gufo.exe` and the `qwen27b` test binaries at current
  HEAD. A host without an XRT install cannot start those binaries at all;
  the `npu-xrt` build directory only carries `xrt_coreutil.lib`.

CPU/core tests (`logit_sampler_test`, `openai_chat_test`, `http_server_test`,
`qwen_chat_template_test`) run with only the first two roots. The launcher
scripts prepend the build output and the HIP bin directory; add the vcpkg and
XRT roots the same way when running test executables straight from a shell:

```
$env:PATH = '<repo>\build\gpu-test;C:\dev\TheRock\build\bin;' +
            'C:\dev\vcpkg\installed\x64-windows\bin;' +
            '<xrt-runtime-dir>;' + $env:PATH
```
