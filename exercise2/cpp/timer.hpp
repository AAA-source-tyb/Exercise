#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>
#include <iostream>
#include <string_view>

template <typename T = std::chrono::milliseconds>
struct Timer {
    using Clock = std::chrono::high_resolution_clock;
    using Duration = T;

    Clock::time_point start_time;

    Timer() : start_time(Clock::now()) {}

    ~Timer() {
        auto end_time = Clock::now();
        auto elapsed = std::chrono::duration_cast<Duration>(end_time - start_time);
        std::cout << "Wall:" << elapsed.count() << time_unit_name<Duration>() << std::endl;
    }

    void reset() {
        start_time = Clock::now();
    }

    T elapsed() const {
        return std::chrono::duration_cast<T>(Clock::now() - start_time);
    }

    friend std::ostream &operator<<(std::ostream &os, const Timer &timer) {
        auto elapsed = timer.elapsed();
        os << "Wall:" << elapsed.count() << time_unit_name<Duration>();
        return os;
    }

    template <typename U>
    static constexpr std::string_view time_unit_name() {
        if constexpr (std::is_same_v<U, std::chrono::hours>) {
            return "h";
        } else if constexpr (std::is_same_v<U, std::chrono::minutes>) {
            return "min";
        } else if constexpr (std::is_same_v<U, std::chrono::seconds>) {
            return "s";
        } else if constexpr (std::is_same_v<U, std::chrono::milliseconds>) {
            return "ms";
        } else if constexpr (std::is_same_v<U, std::chrono::microseconds>) {
            return "us";
        } else if constexpr (std::is_same_v<U, std::chrono::nanoseconds>) {
            return "ns";
        } else {
            return "?";
        }
    }
}; 

#endif 