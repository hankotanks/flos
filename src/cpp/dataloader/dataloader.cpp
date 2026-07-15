#include "dataloader.h"

// system
#include <typeinfo>
#include <fstream>
// package
#include "bag.h"
#include "wild_scenes.h"
// util
#include "util/rosargs.h"

template<typename T>
std::unique_ptr<Dataloader> try_loader(rclcpp::Node* node, std::map<std::string, Dataloader::UnsupportedFormat>& errors) {
    try {
        return std::make_unique<T>(node);
    } catch (const Dataloader::UnsupportedFormat& e) { 
        errors.emplace(typeid(T).name(), e); 
    }
    return nullptr;
}

std::unique_ptr<Dataloader> Dataloader::choose(rclcpp::Node* node) {
    std::map<std::string, Dataloader::UnsupportedFormat> errors;
    if(auto loader = try_loader<BagDataloader>(node, errors)) return loader;
    if(auto loader = try_loader<WildScenesDataloader>(node, errors)) return loader;
    std::ostringstream stream;
    stream << "provided data does not resemble any supported format: ";
    for(const auto& [name, error] : errors) {
        stream << std::endl << "\t" << name << ": " << error.what();
    }
    RCLCPP_ERROR_ONCE(node->get_logger(), "%s", stream.str().c_str());
    rclcpp::shutdown();
    return nullptr;
}

Dataloader::Dataloader(rclcpp::Node* node) {
    util::param_req_path(node, "path_to_data", path_to_data_, true);
}