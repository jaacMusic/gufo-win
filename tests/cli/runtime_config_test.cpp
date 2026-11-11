#include "src/cli/serve/runtime_config.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

#include "src/core/json.hpp"

namespace {

using gufo::json::Value;
using gufo::server::RuntimeConfig;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

}  // namespace

int main() {
  RuntimeConfig config;
  std::size_t max_pending = 4;
  config.Add(
      "max-pending", [&max_pending] { return Value(max_pending); },
      [&max_pending](const Value& value) -> std::string {
        if (!value.is_number() || value.as_size(0) == 0) {
          return "max-pending must be a positive integer";
        }
        max_pending = value.as_size(0);
        return {};
      });

  Expect(config.Snapshot().member_size("max-pending") == 4,
         "snapshots report the live value");

  auto applied = config.Apply(gufo::json::parse(R"({"max-pending": 9})"));
  Expect(max_pending == 9, "a known flag is applied");
  Expect(applied["applied"].size() == 1, "an applied flag is listed once");
  Expect(applied["rejected"].empty(), "an applied flag is not rejected");

  auto mixed =
      config.Apply(gufo::json::parse(R"({"max-pending": 0, "context": 8192})"));
  Expect(max_pending == 9, "a refused value leaves the setting alone");
  Expect(mixed["applied"].empty(), "nothing was applied");
  const auto& rejected = mixed["rejected"].items();
  Expect(rejected.size() == 2, "every refusal is reported");
  Expect(rejected[0].member_str("flag") == "max-pending" &&
             rejected[1].member_str("flag") == "context",
         "refusals keep request order");
  Expect(rejected[0].member_str("reason") ==
             "max-pending must be a positive integer",
         "a setter reason reaches the caller");
  Expect(rejected[1].member_str("reason") ==
             "not runtime-updatable; restart the server with the new flag",
         "an unknown flag asks for a restart");

  std::cout << "runtime_config_test passed\n";
  return 0;
}
