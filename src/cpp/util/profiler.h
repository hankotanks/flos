#pragma once

#include <chrono>
#include <map>
#include <tuple>
#include <iostream>
#include <iomanip>

namespace {

} // private scope

namespace util {
class Profiler {
    using clock = std::chrono::steady_clock;
    using duration = std::chrono::duration<double, std::milli>;
    using instant = std::chrono::steady_clock::time_point;
public:
    Profiler(const std::string& name) : name_(name), start_(Profiler::clock::now()), 
        parent_or_stats_(std::map<std::string, Statistics>{}) { /* STUB */ }
    Profiler(const std::string& name, Profiler* parent) : name_(name), start_(Profiler::clock::now()), 
        parent_or_stats_(parent) { /* STUB */ }
    Profiler() = delete;
    Profiler(const Profiler&) = delete;
    Profiler(Profiler&&) = delete;
    Profiler& operator=(const Profiler&) = delete;
    Profiler& operator=(Profiler&&) = delete;
    // finish the timer
    ~Profiler() {
        double elapsed = Profiler::duration(Profiler::clock::now() - start_).count();
        std::visit([&](auto&& inner) {
            using T = std::decay_t<decltype(inner)>;
            if constexpr (std::is_same_v<T, Profiler*>) {
                // report elapsed time up the stack
                inner->report(name_, elapsed);
            } else if constexpr (std::is_same_v<T, std::map<std::string, Statistics>>) {
                // if its a root Profiler, print stats on completion
                std::cout << name_ << ": " << Statistics{} + elapsed << std::endl;
                for(const auto& [name, stats] : inner) {
                    std::cout << name_ << "/" << name << ": " << stats << std::endl;
                }
            } else static_assert(False<T>);
        }, parent_or_stats_);
    }
private:
    class Statistics {
    public:
        Statistics() = default;
        Statistics(const Statistics&) = delete;
        Statistics(Statistics&&) = default;
        Statistics& operator=(const Statistics&) = delete;
        Statistics& operator=(Statistics&&) = delete;
        Statistics& operator+=(double elapsed) {
            count_++;
            avg_ += (elapsed - avg_) / count_; 
            return *this;
        }
        friend Statistics operator+(Statistics stats, double elapsed) {
            stats += elapsed;
            return stats;
        }
        friend std::ostream& operator<<(std::ostream& os, const Statistics& stats) {
            return os << std::fixed << std::setprecision(2) 
                << stats.avg_ << "ms, " << 1000.0 / stats.avg_ << "Hz "
                << "[" << stats.count_ << " samples]";
        }
    private:
        double avg_{0.0};
        size_t count_{0};    
    };
private:
    void report(const std::string& name, double elapsed) {
        std::visit([&](auto&& inner) {
            using T = std::decay_t<decltype(inner)>;
            if constexpr (std::is_same_v<T, Profiler*>) {
                inner->report(name_ + "/" + name, elapsed);
            } else if constexpr (std::is_same_v<T, std::map<std::string, Statistics>>) {
                if(inner.count(name) == 0)
                    inner.emplace(std::piecewise_construct, std::forward_as_tuple(name), std::forward_as_tuple());
                inner.at(name) += elapsed;
            } else static_assert(Profiler::False<T>);
        }, parent_or_stats_);
    }
private:
    // C++ shenanigans to prevent static_assert from being optimized out of std::visit calls
    template<typename>
    inline static const constexpr bool False = false;

    std::variant<Profiler*, std::map<std::string, Statistics>> parent_or_stats_;
    std::string name_;
    Profiler::instant start_;
};
}