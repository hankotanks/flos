#pragma once

// ros
#include <rclcpp/rclcpp.hpp>

namespace util {
// helper function for range bounded optional parameters
// so I can keep defaults in OdometrySystem::Config directly
#if 0
template<typename T>
static std::optional<T> numeric_param(rclcpp::Node* node, const std::string& name, std::optional<T> to = std::nullopt) {
    static_assert(std::is_integral_v<T> || std::is_floating_point_v<T> || std::is_same_v<T, bool>);
    if constexpr (std::is_same_v<T, bool>) {
        node->declare_parameter<bool>(name);
        rclcpp::Parameter value;
        node->get_parameter(name, value);
        auto overrides = node->get_node_parameters_interface()->get_parameter_overrides();
        if(overrides.find(name) == overrides.end()) return std::nullopt;
        return value.as_bool();
    } else {
        rcl_interfaces::msg::ParameterDescriptor desc{};
        desc.dynamic_typing = true;
        if constexpr (std::is_integral_v<T>) {
            rcl_interfaces::msg::IntegerRange range{};
            range.from_value = 0;
            range.to_value = std::numeric_limits<rcl_interfaces::msg::IntegerRange::_to_value_type>::max();
            desc.integer_range.push_back(range);
        } else if constexpr (std::is_floating_point_v<T>) {
            rcl_interfaces::msg::FloatingPointRange range{};
            range.from_value = 0.0;
            range.to_value = std::numeric_limits<rcl_interfaces::msg::FloatingPointRange::_to_value_type>::max();
            desc.floating_point_range.push_back(range);
        }
        node->declare_parameter(name, rclcpp::ParameterValue{}, desc);
        T value;
        if(node->get_parameter(name, value)) return value;
        return std::nullopt;
    }
}
#endif

// NOTE: This is terrible, but I'm leaving it for now
// C++ can't resolve the type for size_t, because it could be int64_t, unsigned long, etc..
// so I made this map from one type to another
template<typename T>
struct type_map;
template<> struct type_map<size_t> { using to = int64_t; };
template<> struct type_map<float> { using to = float; };
template<> struct type_map<double> { using to = double; };
template<> struct type_map<bool> { using to = bool; };

template<typename T>
static void opt_param(rclcpp::Node* node, T& value, const std::string& name) {
    static_assert(std::is_integral_v<T> || std::is_floating_point_v<T> || std::is_same_v<T, bool>);
    using R = typename type_map<T>::to;
    T temp;
    node->declare_parameter<R>(name, temp);
    node->get_parameter(name, temp);
    const auto& overrides = node->get_node_parameters_interface()->get_parameter_overrides();
    if(overrides.count(name) == 0) return;
    value = temp;
}
}