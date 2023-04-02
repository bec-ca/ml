#pragma once

#include <cstddef>
#include <vector>

namespace ml {

struct DataPoint {
  inline float operator[](int idx) const { return features[idx]; }
  inline float& operator[](int idx) { return features[idx]; }

  size_t size() const { return features.size(); }

  std::vector<float> features;
  double label;
};

} // namespace ml
