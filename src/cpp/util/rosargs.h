#pragma once

// system
#include <optional>
#include <limits>
// ros
#include <rclcpp/rclcpp.hpp>

namespace util {
// helper function for range bounded optional parameters
// so I can keep defaults in OdometrySystem::Config directly
template<typename T>
static std::optional<T> numeric_param(rclcpp::Node* node, const std::string& name, std::optional<T> to = std::nullopt) {
    static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>);
    rcl_interfaces::msg::ParameterDescriptor desc{};
    desc.dynamic_typing = true;
    if constexpr (std::is_integral_v<T>) {
        rcl_interfaces::msg::IntegerRange range{};
        range.from_value = 0;
        range.to_value = to ? static_cast<rcl_interfaces::msg::IntegerRange::_to_value_type>(*to) : \
            std::numeric_limits<rcl_interfaces::msg::IntegerRange::_to_value_type>::max();
        desc.integer_range.push_back(range);
    } else if constexpr (std::is_floating_point_v<T>) {
        rcl_interfaces::msg::FloatingPointRange range{};
        range.from_value = 0.0;
        range.to_value = to ? static_cast<rcl_interfaces::msg::FloatingPointRange::_to_value_type>(*to) : \
            std::numeric_limits<rcl_interfaces::msg::FloatingPointRange::_to_value_type>::max();
        desc.floating_point_range.push_back(range);
    }
    node->declare_parameter(name, rclcpp::ParameterValue{}, desc);
    T value;
    if(node->get_parameter(name, value)) return value;
    return std::nullopt;
}
}