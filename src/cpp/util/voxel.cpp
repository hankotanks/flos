#include "voxel.h"

namespace util {
void voxel_downsample(util::CloudView& cloud, const double voxel_size) {
    // TODO: Could this be static to prevent allocation every frame??
    tsl::robin_map<Eigen::Vector3i, size_t> grid;
    grid.reserve(cloud.size());
    size_t i_write = 0;
    for(size_t i_read = 0; i_read < cloud.size(); ++i_read) {
        const Eigen::Vector3i voxel = point_to_voxel(cloud[i_read].point(), voxel_size);
        if(!grid.contains(voxel)) {
            grid.insert({voxel, i_write});
            cloud[i_write].replace(cloud[i_read]);
            ++i_write;
        }
    }
    cloud.resize(i_write);
}
}