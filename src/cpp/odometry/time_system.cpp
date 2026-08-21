#include "odometry/time_system.h"

// system
#include <iostream>

// i honestly regret adding this pseudo-newtype...
TimeSystem::Stamp::Stamp(unsigned int s, unsigned int ns) : 
    stamp(static_cast<long double>(s) + 1e-9L * static_cast<long double>(ns)),
    sec(s),
    nanosec(ns) { /* STUB */ } 
TimeSystem::Stamp::Stamp(long double s) : 
    stamp(s),
    sec(static_cast<unsigned int>(s)),
    nanosec(static_cast<unsigned int>((s - static_cast<long double>(sec)) * 1e9L)) { /* STUB */ }
TimeSystem::Stamp::Stamp() : stamp(std::numeric_limits<long double>::lowest()) { /* STUB */ }

TimeSystem::MotionPrior TimeSystem::predict_motion(const util::Cloud& points, const Sophus::SE3f& last_delta, TimeSystem::Stamp stamp) {
    auto sweep_period = TimeSystem::period(points);
    // we can still deskew with a constant rate, we just treat the entire dt as sweep_period
    if(stamp.stamp < 0.0) return {last_delta, last_delta, sweep_period, 1.0};
    // by default, if there are no per-point timestamps, 
    // the supplied stamp is assumed to correspond to the end of the scan.
    long double curr_stamp = stamp.stamp;
    if(offset_frame_ == std::numeric_limits<long double>::lowest()) offset_frame_ = stamp.stamp;
    if(sweep_period) {
        if(offset_sweep_ == std::numeric_limits<long double>::lowest()) offset_sweep_ = sweep_period->end;
        // NOTE: we currently assume three possible cases
        // 1. per-point stamps are absolute w.r.t. the same epoch as the frame stamp
        // 2. per-point stamps are relative w.r.t. the frame stamp
        // 3. per-point stamps are absolute w.r.t. a different epoch than the frame stamp
        if(std::abs(offset_sweep_ - offset_frame_) < 1.0) {
            // absolute, same epoch
            curr_stamp = sweep_period->end;
            std::cout << "absolute, same epoch" << std::endl;
        } else if(std::abs(sweep_period->start) < 1.0 && std::abs(sweep_period->end) < 1.0) {
            // relative
            curr_stamp += sweep_period->end;
            std::cout << "relative" << std::endl;
        } else {
            // absolute, but different epoch
            curr_stamp += sweep_period->end - offset_sweep_;
            std::cout << "absolute, different epoch" << std::endl;
        }
    }
    long double prev_dt = prev_dt_;
    Sophus::SE3f frame_delta = last_delta;
    if(prev_stamp_ > 0.0) {
        const long double curr_dt = curr_stamp - prev_stamp_;
        const auto xi = last_delta.cast<long double>().log();
        if(prev_dt > 0.0) frame_delta = Sophus::SE3d::exp((xi * curr_dt / prev_dt).cast<double>()).cast<float>();
        prev_dt_ = curr_dt;
    }
    // compute motion during actual scan if we know its period and deskew
    Sophus::SE3f sweep_delta = frame_delta;
    if(sweep_period && prev_dt > 0.0) {
        const auto xi = frame_delta.cast<long double>().log();
        sweep_delta = Sophus::SE3d::exp((xi * ((sweep_period->end - sweep_period->start) / prev_dt_)).cast<double>()).cast<float>();
    }
    prev_stamp_ = curr_stamp;
    // return the motion prior
    return {frame_delta, sweep_delta, sweep_period, prev_dt_};
}

void TimeSystem::deskew(util::Cloud& points, const MotionPrior& prior) const noexcept {
    if(!prior.sweep_period) return;
    if(auto field = TimeSystem::stamps<float>(points)) {
        TimeSystem::deskew_inner(points, *field, prior.sweep_delta, *(prior.sweep_period));
    } else if(auto field = TimeSystem::stamps<double>(points)) {
        TimeSystem::deskew_inner(points, *field, prior.sweep_delta, *(prior.sweep_period));
    }
}

// field names to search in get_stamp_field
static constexpr std::array<std::string, 4> NAMES = {"timestamps", "time", "timestamp", "t"};

template<typename T>
std::optional<std::span<const T>> TimeSystem::stamps(const util::Cloud& points) {
    for(const std::string& name : NAMES) {
        try { return points.field<T>(name); } catch(...) { continue; }
    }
    return std::nullopt;
}

template std::optional<std::span<const float>> TimeSystem::stamps(const util::Cloud& points);
template std::optional<std::span<const double>> TimeSystem::stamps(const util::Cloud& points);

std::optional<TimeSystem::ScanPeriod> TimeSystem::period(const util::Cloud& points) {
    auto inner = []<typename T>(std::span<const T> field) -> ScanPeriod {
        const auto& [start, end] = std::minmax_element(field.begin(), field.end());
        return ScanPeriod{static_cast<long double>(*start), static_cast<long double>(*end)};
    };
    ScanPeriod period;
    if(auto field = TimeSystem::stamps<float>(points)) {
        return inner.operator()<float>(*field);
    } else if(auto field = TimeSystem::stamps<double>(points)) {
        return inner.operator()<double>(*field);
    } else return std::nullopt;
}

template<typename T>
void TimeSystem::deskew_inner(util::Cloud& points, std::span<const T> stamps, const Sophus::SE3f& sweep_delta, const TimeSystem::ScanPeriod& sweep_period) const noexcept {
    const auto t_norm = [&sweep_period](const T t) -> double { 
        return static_cast<double>(static_cast<long double>(t) - sweep_period.start) / (sweep_period.end - sweep_period.start); 
    };
    const auto& omega = sweep_delta.log().cast<double>();
    tbb::parallel_for(tbb::blocked_range<size_t>{0, points.size()}, [&](const tbb::blocked_range<size_t>& range) {
        for(size_t i = range.begin(); i < range.end(); ++i) {
            const double stamp = t_norm(stamps[i]);
            const auto pose = Sophus::SE3d::exp((stamp - 1.0) * omega).cast<float>();
            points[i] = pose * points[i];
        };
    });
}

template void TimeSystem::deskew_inner(util::Cloud& points, std::span<const float> stamps, const Sophus::SE3f& sweep_delta, const TimeSystem::ScanPeriod& sweep_period) const noexcept;
template void TimeSystem::deskew_inner(util::Cloud& points, std::span<const double> stamps, const Sophus::SE3f& sweep_delta, const TimeSystem::ScanPeriod& sweep_period) const noexcept;