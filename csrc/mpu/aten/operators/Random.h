#pragma once

#include <ATen/core/DistributionsHelper.h>
#include <ATen/core/MT19937RNGEngine.h>
#include <ATen/core/PhiloxRNGEngine.h>
#include <aten/operators/MemoryAccess.h>
#include <utils/DPCPP.h>

#include "comm/Numerics.h"

// TODO: move this into the GeneratorImpl in pytorch-1.7 or later
using Philox4_32_10 = at::Philox4_32;
using mt19937 = at::mt19937;
template <typename engine_t = Philox4_32_10>
class RandomState final {
 public:
  template <
      typename T = engine_t,
      std::enable_if_t<std::is_same<T, Philox4_32_10>::value, int> = 0>
  RandomState(
      uint64_t seed = 67280421310721,
      uint64_t subsequence = 0,
      uint64_t offset = 0)
      : engine(seed, subsequence, offset){};

  template <
      typename T = engine_t,
      std::enable_if_t<std::is_same<T, mt19937>::value, int> = 0>
  RandomState(uint64_t seed = 67280421310721) : engine(seed){};

  // cannot be copied
  RandomState() = delete;
  RandomState(const RandomState&) = delete;
  RandomState& operator=(const RandomState&) = delete;
  RandomState(RandomState&&) = default;
  RandomState& operator=(RandomState&&) = default;

  inline uint64_t make64BitsFrom32Bits(uint32_t hi, uint32_t lo) {
    return (static_cast<uint64_t>(hi) << 32) | lo;
  }

  template <typename T, typename V>
  inline dist_acctype<T> uniform_real(V val, T from, T to) {
    constexpr auto MASK = static_cast<V>(
        (static_cast<uint64_t>(1) << std::numeric_limits<T>::digits) - 1);
    constexpr auto DIVISOR = static_cast<dist_acctype<T>>(1) /
        (static_cast<uint64_t>(1) << std::numeric_limits<T>::digits);
    dist_acctype<T> x = (val & MASK) * DIVISOR;
    return (x * (to - from) + from);
  }

  template <typename T>
  T uniform() {
    if (std::is_same<T, double>::value) {
      uint64_t val = make64BitsFrom32Bits(engine(), engine());
      return uniform_real<T>(val, 0.0, 1.0);
    } else {
      uint32_t val = engine();
      return uniform_real<T>(val, 0.0, 1.0);
    }
  }

  template <typename T, int vec_size>
  at::native::Memory::aligned_vector_loop<T, vec_size> uniform() {
    at::native::Memory::aligned_vector_loop<T, vec_size> result;
#pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      result[i] = uniform<T>();
    }
    return result;
  }

  /**
   * Samples a normal distribution using the Box-Muller method
   * Takes mean and standard deviation as inputs
   * Note that Box-muller method returns two samples at a time.
   * We simply discard one result.
   */
  template <typename T>
  T normal() {
    dist_acctype<T> ret;
    dist_acctype<T> u1 = uniform<dist_acctype<T>>();
    dist_acctype<T> u2 = uniform<dist_acctype<T>>();
    const dist_acctype<T> r = Numerics<dist_acctype<T>>::sqrt(
        static_cast<dist_acctype<T>>(-2.0) *
        Numerics<dist_acctype<T>>::log(static_cast<dist_acctype<T>>(1.0) - u2));
    const dist_acctype<T> theta = static_cast<dist_acctype<T>>(2.0) *
        Numerics<dist_acctype<T>>::pi() * u1;
    ret = r * Numerics<dist_acctype<T>>::cos(theta);
    return static_cast<T>(ret);
  }

  template <typename T>
  T random() {
    if (std::is_same<T, uint64_t>::value) {
      return make64BitsFrom32Bits(engine(), engine());
    } else {
      return engine();
    }
  }

 private:
  engine_t engine;
};
