#pragma once

// ros
#include <rclcpp/rclcpp.hpp>
// package
#include "dataloader/dataloader.h"
// util
#include "util/rosbag.h"

class BagDataloader : public Dataloader {
    friend class Dataloader;
public:
    explicit BagDataloader(rclcpp::Node* node);
    BagDataloader(const BagDataloader&) = delete;
    BagDataloader(BagDataloader&&) = delete;
    BagDataloader& operator=(const BagDataloader&) = delete;
    BagDataloader& operator=(BagDataloader&&) = delete;
    ~BagDataloader();
public:
    virtual size_t init(const std::shared_ptr<rclcpp::Node>& node) override final;
    virtual bool finished() override final;
    virtual std::pair<util::Cloud, std_msgs::msg::Header> next() override final;
private:
    std::filesystem::path path_to_bag_;
    std::string incoming_points_topic_;
    std::unique_ptr<util::BufferableBag> bag_;
};