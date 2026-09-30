#pragma once

// ros
#include <rclcpp/rclcpp.hpp>
#include <stdexcept>
// util
#include "util/cloud.h"

class Dataloader {
public:
    struct UnsupportedFormat : std::runtime_error {
        using std::runtime_error::runtime_error;
    };
public:
    static std::unique_ptr<Dataloader> choose(rclcpp::Node* node);
    explicit Dataloader(rclcpp::Node* node);
    Dataloader(const Dataloader&) = delete;
    Dataloader(Dataloader&&) = delete;
    Dataloader& operator=(const Dataloader&) = delete;
    Dataloader& operator=(Dataloader&&) = delete;
    ~Dataloader() = default;
public:
    virtual size_t init(const std::shared_ptr<rclcpp::Node>& node) = 0;
    virtual bool finished() = 0;
    virtual std::pair<util::Cloud, std_msgs::msg::Header> next() = 0;
protected:
    std::filesystem::path path_to_data_;
};
