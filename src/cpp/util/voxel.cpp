#include "voxel.h"

namespace util {
// TODO: I can avoid another cloud allocation if I implement Point::remove and a reverse Iterator for Cloud
void voxel_downsample(util::CloudView& cloud, const double voxel_size) {
    tsl::robin_map<Eigen::Vector3i, size_t> grid;
    grid.reserve(cloud.size());
    std::span<const Eigen::Vector3f> points = cloud.points();
    for(size_t i = 0; i < points.size(); ++i) {
        const Eigen::Vector3i voxel = point_to_voxel(points[i], voxel_size);
        if(!grid.contains(voxel)) grid.insert({voxel, i});
    }
    util::CloudView cloud_original{std::move(cloud)};
    std::for_each(grid.cbegin(), grid.cend(), [&](const auto& voxel_and_index) {
        cloud.add_point(cloud_original[voxel_and_index.second]);
    });
}
}