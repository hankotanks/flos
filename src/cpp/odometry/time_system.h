#pragma once

// system 
#include <optional>
#include <limits>
// sophus
#include <sophus/se3.hpp>
// tbb
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
// util
#include "util/cloud.h"

class TimeSystem {
public:
    TimeSystem() = default;
    TimeSystem(const TimeSystem&) = delete;
    TimeSystem(TimeSystem&&) = delete;
    TimeSystem& operator=(const TimeSystem&) = delete;
    TimeSystem& operator=(TimeSystem&&) = delete;
    ~TimeSystem() = default;
public:
    struct Stamp {
        explicit Stamp(unsigned int s, unsigned int ns);
        explicit Stamp(long double s);
        explicit Stamp();
        // combined timestamp
        long double stamp;
        // integer representation
        unsigned int sec;
        unsigned int nanosec;
    };
public:
    // represents the duration of the scan sweep (independent of the per-frame timestamp)
    struct ScanPeriod { long double start, end; };
    // frame_delta is the previous motion extrapolated for the entire frame
    // sweep_delta is the extrapolated motion w.r.t. the actual scan period
    struct MotionPrior {
        Sophus::SE3f frame_delta;
        Sophus::SE3f sweep_delta;
        std::optional<ScanPeriod> sweep_period;
        long double dt;
    };
    // get the motion prediction based on the previous delta
    MotionPrior predict_motion(const util::Cloud& points, const Sophus::SE3f& last_delta, Stamp stamp);
    // perform deskewing
    void deskew(util::Cloud& points, const MotionPrior& prior) const noexcept;
private:
    // retrive the timestamp field
    template<typename T>
    static std::optional<std::span<const T>> stamps(const util::Cloud& points);
    // get the scan period
    static std::optional<ScanPeriod> period(const util::Cloud& points);
    // since there is type ambiguity, we have to split deskew
    template<typename T>
    void deskew_inner(util::Cloud& points, std::span<const T> stamps, const Sophus::SE3f& sweep_delta, const ScanPeriod& sweep_period) const noexcept;
private:
    long double offset_frame_{std::numeric_limits<long double>::lowest()};
    long double offset_sweep_{std::numeric_limits<long double>::lowest()};
private:
    long double prev_dt_{std::numeric_limits<long double>::lowest()};
    long double prev_stamp_{std::numeric_limits<long double>::lowest()};
};