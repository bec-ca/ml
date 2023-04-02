#pragma once

#include <cstdint>
#include <limits>
#include <memory>

namespace ml {

struct Rng {
 public:
  using result_type = uint32_t;

  static constexpr result_type min()
  {
    return std::numeric_limits<result_type>::min();
  }

  static constexpr result_type max()
  {
    return std::numeric_limits<result_type>::max();
  }

  virtual ~Rng();

  virtual uint32_t operator()() = 0;

  static std::unique_ptr<Rng> create(uint32_t seed);
};

} // namespace ml
