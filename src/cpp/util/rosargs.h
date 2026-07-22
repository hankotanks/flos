#pragma once

// system
#include <filesystem>
// ros
#include <rclcpp/rclcpp.hpp>

namespace util {
template<typename T>
struct is_std_vector : std::false_type {};

template<typename T, typename Alloc>
struct is_std_vector<std::vector<T, Alloc>> : std::true_type {};

template<typename T>
struct is_supported_parameter_type : std::bool_constant<std::is_same_v<T, bool> || std::is_integral_v<T> || std::is_floating_point_v<T> || std::is_same_v<T, std::string> || is_std_vector<T>::value> { /* STUB */ };

template<typename T>
bool param_opt(rclcpp::Node* node, const std::string& name, T& value) noexcept {
    static_assert(is_supported_parameter_type<T>::value, "unsupported parameter type");
    if(!node->has_parameter(name)) {
        if constexpr (std::is_same_v<T, std::size_t>) {
            node->declare_parameter<int64_t>(name, static_cast<int64_t>(value));
        }  else if constexpr (std::is_same_v<T, float>) {
            node->declare_parameter<double>(name, static_cast<double>(value));
        } else if constexpr (is_std_vector<T>::value) {
            if constexpr (std::is_same_v<typename T::value_type, std::size_t>) {
                std::vector<int64_t> value_int64;
                value_int64.reserve(value.size());
                std::transform(value.begin(), value.end(),
                    std::back_inserter(value_int64), [](size_t val) { return static_cast<int64_t>(val); });
                node->declare_parameter<std::vector<int64_t>>(name, value_int64);
            } else if constexpr (std::is_same_v<typename T::value_type, float>) {
                std::vector<double> value_double;
                value_double.reserve(value.size());
                std::transform(value.begin(), value.end(),
                    std::back_inserter(value_double), [](size_t val) { return static_cast<double>(val); });
                node->declare_parameter<std::vector<double>>(name, value_double);
            } else {
                node->declare_parameter<T>(name, value);
            }
        } else {
            node->declare_parameter<T>(name, value);
        }
    }
    const auto& overrides = node->get_node_parameters_interface()->get_parameter_overrides();
    if(overrides.find(name) != overrides.end()) {
        if constexpr (std::is_same_v<T, std::size_t>) {
            int64_t value_int64;
            node->get_parameter(name, value_int64);
            if(value_int64 < 0) {
                RCLCPP_ERROR_STREAM_ONCE(node->get_logger(), name << " must be positive: " << value_int64);
                rclcpp::shutdown();
            }
            value = static_cast<size_t>(value_int64);
        } else if constexpr (std::is_same_v<T, float>) {
            double value_double;
            node->get_parameter(name, value_double);
            value = static_cast<float>(value_double);
        } else if constexpr (is_std_vector<T>::value) {
            if constexpr (std::is_same_v<typename T::value_type, std::size_t>) {
                std::vector<int64_t> value_int64;
                node->get_parameter(name, value_int64);
                value.clear();
                value.reserve(value_int64.size());
                std::transform(value_int64.begin(), value_int64.end(),
                    std::back_inserter(value), [node, &name](int64_t val) { 
                        if(val < 0) {
                            RCLCPP_ERROR_STREAM_ONCE(node->get_logger(), "all " << name << " values must be positive, but contained: " << val);
                            rclcpp::shutdown();
                        }
                        return static_cast<size_t>(val); 
                    });
            } else if constexpr (std::is_same_v<typename T::value_type, float>) { 
                std::vector<double> value_double;
                node->get_parameter(name, value_double);
                value.clear();
                value.reserve(value_double.size());
                std::transform(value_double.begin(), value_double.end(),
                    std::back_inserter(value), [node, &name](double val) { return static_cast<float>(val); });
            } else {
                node->get_parameter(name, value);
            }
        } else {
            node->get_parameter(name, value);
        }
        return true;
    }
    return false;
}

template<typename T>
void param_req(rclcpp::Node* node, const std::string& name, T& value) noexcept {
    if(!param_opt(node, name, value)) {
        RCLCPP_ERROR_STREAM_ONCE(node->get_logger(), "missing a required argument: " << name);
        rclcpp::shutdown();
    }
}

template<typename T>
bool param_opt_set(rclcpp::Node* node, const std::string& name, std::set<T>& value) {
    std::vector<T> keys;
    util::param_opt(node, name, keys);
    if(keys.empty()) return false;
    value.clear();
    value.insert(keys.begin(), keys.end());
    return true;
}

template<typename T>
void param_req_set(rclcpp::Node* node, const std::string& name, std::set<T>& value) {
    if(!param_opt_set(node, name, value)) {
        RCLCPP_ERROR_STREAM_ONCE(node->get_logger(), "missing a required argument: " << name);
        rclcpp::shutdown();
    }
}

template<typename K, typename V>
bool param_opt_map(rclcpp::Node* node, const std::string& name, std::map<K, V>& value) {
    std::vector<K> keys;
    param_opt<std::vector<K>>(node, name + "_key", keys);
    std::vector<V> vals;
    param_opt<std::vector<V>>(node, name + "_val", vals);
    if(keys.size() != vals.size()) {
        RCLCPP_ERROR_STREAM_ONCE(node->get_logger(), name << " requires an equal number of keys and values");
        rclcpp::shutdown();
    }
    if(keys.empty()) return false; 
    for(size_t i = 0; i < keys.size(); ++i) value.emplace(keys[i], vals[i]);
    return true;
}

template<typename K, typename V>
void param_req_map(rclcpp::Node* node, const std::string& name, std::map<K, V>& value) {
    if(!param_opt_map(node, name, value)) {
        RCLCPP_ERROR_STREAM_ONCE(node->get_logger(), "missing a required argument: " << name);
        rclcpp::shutdown();
    }
}

bool param_opt_path(rclcpp::Node* node, const std::string& name, std::filesystem::path& value, bool exists = false, std::optional<bool> file = std::nullopt, bool create_directories = false);

void param_req_path(rclcpp::Node* node, const std::string& name, std::filesystem::path& value, bool exists = false, std::optional<bool> file = std::nullopt, bool create_directories = false);
}