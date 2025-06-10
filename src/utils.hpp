#pragma once

#include <chrono>
#include "types.hpp"

inline auto time_point_now()
{
    return std::chrono::steady_clock::now();
}

constexpr f64 span_ms(const auto& start, const auto& end)
{
    return std::chrono::duration_cast<std::chrono::duration<f64, std::micro>>(end - start).count() / 1000.0;
}

constexpr f64 span_us(const auto& start, const auto& end)
{
    return std::chrono::duration_cast<std::chrono::duration<f64, std::micro>>(end - start).count();
}

inline void* malloc_aligned(const size_t alignment_, size_t buffer_size)
{
    if (buffer_size == 0) return nullptr;
    buffer_size = (buffer_size + (alignment_ - 1)) & ~(alignment_ - 1);
#ifdef _WIN32
    return _aligned_malloc(buffer_size, alignment_);
#else
   return std::aligned_alloc(alignment_, buffer_size);
#endif // 
}

inline void free_aligned(void* ptr) noexcept
{
    if (ptr == nullptr) return;
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}
