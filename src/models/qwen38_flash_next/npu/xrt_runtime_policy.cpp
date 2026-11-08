#include <stdexcept>

#include "xrt_runtime.hpp"

namespace gufo::npu::detail {

// Adapted from the XRT host flow; build/code are unsigned
// integers, not a Git hash. No device API belongs in this loading boundary.
void ValidateAndLoad(RuntimeOperations& operations) {
  const auto identity = operations.Inspect();
  if (!identity.path.is_absolute() ||
      identity.path.filename() != "xrt_coreutil.dll") {
    throw std::runtime_error("NPU XRT requires an absolute runtime DLL path");
  }
  // Re-certified 2026-11 against the WHQL-signed KMD 32.00.20102.4481
  // System32 drop; the XRT host ABI below (2.21.0 build 8679) is unchanged.
  if (identity.sha256 !=
      "d0dce13280a0d668b497d4fcabdcb43f1a7353e466af3acabd98c6df7607ed8e") {
    throw std::runtime_error("NPU XRT DLL SHA256 mismatch");
  }
  if (identity.file_version != std::array<unsigned, 4>{32, 0, 20102, 4481}) {
    throw std::runtime_error("NPU XRT DLL file version mismatch");
  }
  operations.Load(identity.path);
  if (operations.Versions() !=
      std::array<unsigned, 5>{2, 21, 0, 8679, 131093}) {
    throw std::runtime_error("NPU XRT five-function version mismatch");
  }
}

}  // namespace gufo::npu::detail
