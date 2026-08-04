#pragma once

// system
#include <chrono>
#include <map>
#include <tuple>
#include <iostream>
#include <iomanip>

// helper macro for treating the profiler a block wrapper
#ifndef PROFILER_DISABLE
#define PROFILE(name, parent) \
    for(bool _once_##name = true; _once_##name; _once_##name = false) \
        for(util::Profiler name{#name, parent}; _once_##name; _once_##name = false)
#else
#define PROFILE(name, parent) 
#endif // PROFILER_DISABLE

#ifndef PROFILER_DISABLE
#define PROFILER_CONCAT_IMPL(a,b) a##b
#define PROFILER_CONCAT(a,b) PROFILER_CONCAT_IMPL(a,b)
#define PROFILE_SKIP(name) \
    for(bool PROFILER_CONCAT(_once_, __LINE__) = true; \
        PROFILER_CONCAT(_once_, __LINE__); \
        PROFILER_CONCAT(_once_, __LINE__) = false) \
        for(util::Profiler::Skip PROFILER_CONCAT(_skip_, __LINE__){name}; \
            PROFILER_CONCAT(_once_, __LINE__); \
            PROFILER_CONCAT(_once_, __LINE__) = false)
#else
#define PROFILE_SKIP(name)
#endif // PROFILER_DISABLE

namespace util {
#ifndef PROFILER_DISABLE
class Profiler {
    using clock = std::chrono::steady_clock;
    using duration = std::chrono::duration<double, std::milli>;
    using instant = std::chrono::steady_clock::time_point;
public:
    class Skip;
    friend class Skip;
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
        if (pause_ > 0) {
            excluded_ += duration(clock::now() - pause_start_).count();
            pause_ = 0;
            if(auto ptr = std::get_if<Profiler*>(&parent_or_stats_)) (*ptr)->resume();
        }
        double elapsed = Profiler::duration(Profiler::clock::now() - start_).count() - excluded_;
        std::visit([&](auto&& inner) {
            using T = std::decay_t<decltype(inner)>;
            if constexpr (std::is_same_v<T, Profiler*>) {
                // report elapsed time up the stack
                if(inner != nullptr) inner->report(name_, elapsed);
            } else if constexpr (std::is_same_v<T, std::map<std::string, Statistics>>) {
                // if its a root Profiler, print stats on completion
                std::cout << name_ << ": " << Statistics{} + elapsed << std::endl;
                for(const auto& [name, stats] : inner) {
                    std::cout << name_ << "/" << name << ": " << stats << std::endl;
                }
            } else static_assert(False<T>);
        }, parent_or_stats_);
    }
public:
    class Skip {
    public:
        explicit Skip(Profiler& profiler) : profiler_(profiler) { profiler_.pause(); }
        Skip(const Skip&) = delete;
        Skip(Skip&&) = delete;
        Skip& operator=(const Skip&) = delete;
        Skip& operator=(Skip&&) = delete;
        ~Skip() { profiler_.resume(); }
    private:
        Profiler& profiler_;
    };
private:
    void pause() {
        if(pause_++ == 0) {
            pause_start_ = clock::now();
            if(auto ptr = std::get_if<Profiler*>(&parent_or_stats_)) {
                (*ptr)->pause();
            }
        }
    }
    void resume() {
        if(pause_ == 0) return;
        if(--pause_ == 0) {
            excluded_ += duration(clock::now() - pause_start_).count();
            if(auto ptr = std::get_if<Profiler*>(&parent_or_stats_)) {
                (*ptr)->resume();
            }
        }
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
                if(inner != nullptr) inner->report(name_ + "/" + name, elapsed);
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
    // state for tracking excluded time blocks
    size_t pause_{0};
    instant pause_start_{};
    double excluded_{0.0};
};
#else
class Profiler {
public:
    Profiler(const std::string&) { /* STUB */ }
    Profiler(const std::string&, Profiler*) { /* STUB */ }
    Profiler() = delete;
    Profiler(const Profiler&) = delete;
    Profiler(Profiler&&) = delete;
    Profiler& operator=(const Profiler&) = delete;
    Profiler& operator=(Profiler&&) = delete;
    // finish the timer
    ~Profiler() = default;
};
#endif // PROFILER_DISABLE
}