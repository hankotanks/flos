#pragma once

#include "util/cloud.h"

// system
#include <numeric>

#ifdef GENERATE_CLOUD_PYBIND_MODULE
using FieldI1 = std::vector<int8_t>;
using FieldU1 = std::vector<uint8_t>;
using FieldI2 = std::vector<int16_t>;
using FieldU2 = std::vector<uint16_t>;
using FieldI4 = std::vector<int32_t>;
using FieldU4 = std::vector<uint32_t>;
using FieldF4 = std::vector<float>;
using FieldF8 = std::vector<double>;

PYBIND11_MAKE_OPAQUE(FieldI1);
PYBIND11_MAKE_OPAQUE(FieldU1);
PYBIND11_MAKE_OPAQUE(FieldI2);
PYBIND11_MAKE_OPAQUE(FieldU2);
PYBIND11_MAKE_OPAQUE(FieldI4);
PYBIND11_MAKE_OPAQUE(FieldU4);
PYBIND11_MAKE_OPAQUE(FieldF4);
PYBIND11_MAKE_OPAQUE(FieldF8);
PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3f>);
#endif // GENERATE_CLOUD_PYBIND_MODULE

namespace {
// this algorithm is sourced from Potatoswatter on StackOverflow: https://stackoverflow.com/a/1267878
template<typename order_iterator, typename value_iterator>
void reorder(order_iterator order_begin, order_iterator order_end, value_iterator v)  {   
    typedef typename std::iterator_traits<value_iterator>::value_type value_t;
    typedef typename std::iterator_traits<order_iterator>::value_type index_t;
    typedef typename std::iterator_traits<order_iterator>::difference_type diff_t;
    
    diff_t remaining = order_end - 1 - order_begin;
    for(index_t s = index_t(), d; remaining > 0; ++s) {
        for(d = order_begin[s]; d > s; d = order_begin[d]);
        if(d == s) {
            --remaining;
            value_t temp = v[s];
            while(d = order_begin[d], d != s) {
                std::swap(temp, v[d]);
                --remaining;
            }
            v[s] = temp;
        }
    }
}
} // INTERNAL

namespace util {
template<typename T, typename Predicate>
void Cloud::filter(const std::string& name, Predicate&& pred) {
    std::span<const T> field = Cloud::field<T>(name);
    size_t i_write = 0;
    for (size_t i_read = 0; i_read < Cloud::size(); ++i_read) {
        if(!pred(field[i_read])) continue;
        if(i_write != i_read) {
            points_[i_write] = points_[i_read];
            for(Field& f : fields_) {
                std::visit([&](auto& vec) { vec[i_write] = vec[i_read]; }, f.data_);
            }
        }
        ++i_write;
    }
    Cloud::resize(i_write);
}

template<typename T, typename Predicate>
void Cloud::filter_by_distance(const Eigen::Vector3f& origin, float minima, float maxima, const std::string& name, Predicate&& pred) {
    float minima_sq = minima * minima;
    float maxima_sq = maxima * maxima;
    std::span<const T> field = Cloud::field<T>(name);
    const size_t n = points_.size();
    size_t i_write = 0;
    for(size_t i_read = 0; i_read < n; ++i_read) {
        bool keep = true;
        if(pred(field[i_read])) {
            const float d_sq = (points_[i_read] - origin).squaredNorm();
            keep = (d_sq >= minima_sq && d_sq <= maxima_sq);
        }
        if(!keep) continue;
        if(i_write != i_read) {
            points_[i_write] = points_[i_read];
            for(Field& f : fields_) {
                std::visit([i_write, i_read](auto& vec) { vec[i_write] = vec[i_read]; }, f.data_);
            }
        }
        ++i_write;
    }
    Cloud::resize(i_write);
}

template<typename T, typename Comparator>
void Cloud::sort(const std::string& name, Comparator&& comp) {
    std::span<const T> scalars = Cloud::field<T>(name);
    // build a list of sorted indices
    std::vector<size_t> indices(Cloud::size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&scalars, &comp](size_t fst, size_t snd) {
        return (comp)(scalars[fst], scalars[snd]);
    });
    // reorder points
    reorder(indices.begin(), indices.end(), points_.begin());
    // reorder fields
    for(Field& field : fields_) {
        std::visit([&indices](auto& vec) { 
            reorder(indices.begin(), indices.end(), vec.begin());
        }, field.data_);
    }
}

template<typename T, typename Transform>
void Cloud::transform(const std::string& name, Transform&& transform) {
    std::span<T> scalars = Cloud::field<T>(name);
    std::transform(scalars.begin(), scalars.end(), scalars.begin(), [transform](const T& scalar) {
        return transform(scalar);
    });
}
}
