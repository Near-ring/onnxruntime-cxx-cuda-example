//
// Created by semigroup on 1/12/25.
//
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <execution>
#include <filesystem>
#include <functional>
#include <future>
#include <gsl/narrow>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <ranges>
#include <set>
#include <shared_mutex>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <deque>
#include <queue>

#ifdef _WIN32
#include <intrin.h>
#endif // _WIN32

#include <omp.h>
#include <Eigen/Dense>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/hal/hal.hpp>
#include <opencv2/imgproc/hal/hal.hpp>

#define FMT_UNICODE 0
#include <spdlog/spdlog.h>