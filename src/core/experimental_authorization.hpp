// SPDX-License-Identifier: MIT
#ifndef GUFO_CORE_EXPERIMENTAL_AUTHORIZATION_HPP_
#define GUFO_CORE_EXPERIMENTAL_AUTHORIZATION_HPP_

#include <stdexcept>
#include <string>
#include <utility>

namespace gufo::core {
enum class ExperimentalFeature { Prefill };

// A value capability: default construction is denied. Only the operator opt-in
// (GUFO_NPU_PREFILL=1) grants it, so no code path reaches the NPU by accident.
// Ported from bubo src/core/experimental_authorization.hpp; bubo's release
// resolver is replaced by Granted(), called once from engine startup.
class ExperimentalAuthorization {
public:
  ExperimentalAuthorization() = default;
  bool Allows(ExperimentalFeature feature) const noexcept {
    return allowed_ && feature_ == feature;
  }
  const std::string& reason() const noexcept { return reason_; }
  void Require(ExperimentalFeature feature) const {
    if (!Allows(feature))
      throw std::invalid_argument("experimental mode denied: " + reason_);
  }

  static ExperimentalAuthorization Granted(ExperimentalFeature feature,
                                           std::string reason) {
    return {feature, true, std::move(reason)};
  }

#ifdef GUFO_NPU_TESTING
  // Fake-XRT targets only. Never compiled into the product/counter host.
  static ExperimentalAuthorization FakePrefill() {
    return {ExperimentalFeature::Prefill, true, "fake-XRT test"};
  }
#endif

private:
  ExperimentalAuthorization(ExperimentalFeature feature, bool allowed,
                            std::string reason)
      : feature_(feature), allowed_(allowed), reason_(std::move(reason)) {}
  ExperimentalFeature feature_ = ExperimentalFeature::Prefill;
  bool allowed_ = false;
  std::string reason_ = "NPU policy disabled";
};

// The startup and CPU counter share this creation boundary. The callback is
// never evaluated for a denied capability.
template<class Factory>
bool CreateAuthorizedPrefill(const ExperimentalAuthorization& authorization,
                             Factory&& factory) {
  if (!authorization.Allows(ExperimentalFeature::Prefill))
    return false;
  std::forward<Factory>(factory)(authorization);
  return true;
}
}  // namespace gufo::core
#endif  // GUFO_CORE_EXPERIMENTAL_AUTHORIZATION_HPP_
