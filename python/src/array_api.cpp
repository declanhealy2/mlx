// Copyright © 2026 Apple Inc.

#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "mlx/device.h"
#include "mlx/utils.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

namespace {

struct ArrayNamespaceInfo {};

void validate_device(const std::optional<mx::Device>& device) {
  if (!device.has_value()) {
    return;
  }
  const auto& d = device.value();
  if (d.type == mx::Device::cpu) {
    if (d.index != 0) {
      throw std::invalid_argument("[__array_namespace_info__] Invalid CPU device.");
    }
    return;
  }
  if (d.index < 0 || d.index >= mx::device_count(mx::Device::gpu)) {
    throw std::invalid_argument("[__array_namespace_info__] Invalid GPU device.");
  }
}

bool is_gpu(const std::optional<mx::Device>& device) {
  return device.has_value() && device->type == mx::Device::gpu;
}

const std::vector<std::pair<std::string, mx::Dtype>>& standard_dtypes() {
  static const std::vector<std::pair<std::string, mx::Dtype>> dtypes = {
      {"bool", mx::bool_},
      {"int8", mx::int8},
      {"int16", mx::int16},
      {"int32", mx::int32},
      {"int64", mx::int64},
      {"uint8", mx::uint8},
      {"uint16", mx::uint16},
      {"uint32", mx::uint32},
      {"uint64", mx::uint64},
      {"float16", mx::float16},
      {"float32", mx::float32},
      {"float64", mx::float64},
      {"complex64", mx::complex64},
  };
  return dtypes;
}

bool matches_kind(const mx::Dtype& dtype, const nb::handle& kind) {
  if (kind.is_none()) {
    return true;
  }
  if (nb::isinstance<nb::tuple>(kind)) {
    for (auto item : nb::cast<nb::tuple>(kind)) {
      if (matches_kind(dtype, item)) {
        return true;
      }
    }
    return false;
  }
  const auto value = nb::cast<std::string>(kind);
  if (value == "bool") {
    return dtype == mx::bool_;
  }
  if (value == "signed integer") {
    return mx::issubdtype(dtype, mx::signedinteger);
  }
  if (value == "unsigned integer") {
    return mx::issubdtype(dtype, mx::unsignedinteger);
  }
  if (value == "integral") {
    return mx::issubdtype(dtype, mx::integer);
  }
  if (value == "real floating") {
    return mx::issubdtype(dtype, mx::floating);
  }
  if (value == "complex floating") {
    return mx::issubdtype(dtype, mx::complexfloating);
  }
  if (value == "numeric") {
    return mx::issubdtype(dtype, mx::number);
  }
  throw std::invalid_argument(
      "[__array_namespace_info__.dtypes] Unsupported dtype kind: " + value);
}

} // namespace

void init_array_api(nb::module_& m) {
  nb::class_<ArrayNamespaceInfo>(
      m,
      "__array_namespace_info__",
      R"pbdoc(
      Array API namespace inspection utilities.

      See the Python Array API standard inspection interface for details.
      )pbdoc")
      .def(nb::init<>())
      .def(
          "capabilities",
          [](const ArrayNamespaceInfo&) {
            nb::dict result;
            result["boolean indexing"] = false;
            result["data-dependent shapes"] = false;
            result["max dimensions"] = nb::none();
            return result;
          })
      .def(
          "default_device",
          [](const ArrayNamespaceInfo&) { return mx::default_device(); })
      .def(
          "default_dtypes",
          [](const ArrayNamespaceInfo&,
             const std::optional<mx::Device>& device) {
            validate_device(device);
            nb::dict result;
            result["real floating"] = nb::cast(mx::float32);
            result["complex floating"] = nb::cast(mx::complex64);
            result["integral"] = nb::cast(mx::int32);
            result["indexing"] = nb::cast(mx::int32);
            return result;
          },
          nb::kw_only(),
          "device"_a = nb::none())
      .def(
          "dtypes",
          [](const ArrayNamespaceInfo&,
             const std::optional<mx::Device>& device,
             const nb::object& kind) {
            validate_device(device);
            nb::dict result;
            for (const auto& [name, dtype] : standard_dtypes()) {
              if (is_gpu(device) && dtype == mx::float64) {
                continue;
              }
              if (matches_kind(dtype, kind)) {
                result[nb::str(name.c_str())] = nb::cast(dtype);
              }
            }
            return result;
          },
          nb::kw_only(),
          "device"_a = nb::none(),
          "kind"_a = nb::none())
      .def(
          "devices",
          [](const ArrayNamespaceInfo&) {
            nb::list devices;
            devices.append(mx::Device(mx::Device::cpu, 0));
            const auto gpu_count = mx::device_count(mx::Device::gpu);
            for (int index = 0; index < gpu_count; ++index) {
              devices.append(mx::Device(mx::Device::gpu, index));
            }
            return nb::module_::import_("builtins").attr("tuple")(devices);
          });
}
