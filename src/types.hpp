#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#define _loop while (true)

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef float f32;
typedef double f64;

template<typename T>
using Vec = std::vector<T>;

template<class Key, class Value>
using HashMap = std::unordered_map<Key, Value>;

template<class Key, class Value>
using BSTMap = std::map<Key, Value>;

//========Result and Option========

template <class T>
using Option = std::optional<T>;

template <class T>
inline constexpr auto Some(T&& v) -> Option<T>
{
    return std::make_optional(std::forward<T>(v));
}

inline constexpr std::nullopt_t None = std::nullopt;

using std::array;

template<typename... Args>
using tup = std::tuple<Args...>;

using String = std::string;

#ifndef _WIN32
#define _PURE_FUNC [[gnu::pure]]
#else
#define _PURE_FUNC
#endif //  _WIN32
