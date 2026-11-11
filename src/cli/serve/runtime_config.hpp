#ifndef GUFO_SERVER_RUNTIME_CONFIG_HPP_
#define GUFO_SERVER_RUNTIME_CONFIG_HPP_

#include <algorithm>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "src/core/json.hpp"

namespace gufo::server {

/// Server flag values that can change without restarting the process.
///
/// Each entry pairs a getter with a setter that returns an empty string when it
/// applied the value and a human-readable reason when it refused. Flags that
/// were not registered here (model, context, sockets, ...) are rejected so a
/// client learns it must restart the server.
class RuntimeConfig {
public:
  using Getter = std::function<json::Value()>;
  using Setter = std::function<std::string(const json::Value&)>;

  void Add(std::string name, Getter getter, Setter setter) {
    entries_.push_back({std::move(name), std::move(getter), std::move(setter)});
  }

  [[nodiscard]] json::Value Snapshot() const {
    json::Value values = json::Value::object();
    for (const auto& entry : entries_) {
      values[entry.name] = entry.get();
    }
    return values;
  }

  /// Applies an object of flag name to value and reports every outcome.
  [[nodiscard]] json::Value Apply(const json::Value& updates) const {
    json::Value applied = json::Value::array();
    json::Value rejected = json::Value::array();
    for (const auto& [name, value] : updates.members()) {
      const auto entry = std::find_if(
          entries_.begin(), entries_.end(),
          [&](const Entry& candidate) { return candidate.name == name; });
      std::string reason =
          entry == entries_.end()
              ? "not runtime-updatable; restart the server with the new flag"
              : entry->set(value);
      if (reason.empty()) {
        applied.push_back(name);
        continue;
      }
      json::Value item = json::Value::object();
      item["flag"] = name;
      item["reason"] = std::move(reason);
      rejected.push_back(std::move(item));
    }
    json::Value result = json::Value::object();
    result["applied"] = std::move(applied);
    result["rejected"] = std::move(rejected);
    return result;
  }

private:
  struct Entry {
    std::string name;
    Getter get;
    Setter set;
  };

  std::vector<Entry> entries_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_RUNTIME_CONFIG_HPP_
