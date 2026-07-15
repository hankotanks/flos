#pragma once

// system
#include <filesystem>
// ros
#include <rclcpp/rclcpp.hpp>
// package
#include "dataloader.h"

class WildScenesDataloader : public Dataloader {
    friend class Dataloader;
public:
    explicit WildScenesDataloader(rclcpp::Node* node);
    WildScenesDataloader(const WildScenesDataloader&) = delete;
    WildScenesDataloader(WildScenesDataloader&&) = delete;
    WildScenesDataloader& operator=(const WildScenesDataloader&) = delete;
    WildScenesDataloader& operator=(WildScenesDataloader&&) = delete;
    ~WildScenesDataloader() = default;
public:
    virtual size_t init(const std::shared_ptr<rclcpp::Node>& node) override final;
    virtual bool finished() override final;
    virtual std::pair<util::Cloud, std_msgs::msg::Header> next() override final;
private:
    static constexpr std::string LIDAR_FRAME{"lidar_frame"};
    std::vector<std::filesystem::path> scans_;
    size_t index_{0};
};