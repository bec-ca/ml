#pragma once

#include <cstdint>
#include <vector>

#include "node.hpp"

namespace ml {

struct FastNode {
  float v;
  int32_t feature;
  int32_t idx;

  static FastNode make_leaf(float value, int32_t next_tree);
  static FastNode make_inner(int32_t feature, float threshold);
};

struct FastTree {
 public:
  FastTree(const std::vector<Node>& node);

  FastTree(const FastTree& node);
  FastTree(FastTree&& node);

  ~FastTree();

  template <class T> float eval(const T& features) const
  {
    int32_t idx = 0;
    int32_t size = std::ssize(_nodes);
    float sum = 0;
    while (idx < size) {
      while (_nodes[idx].feature != -1) {
        idx +=
          features[_nodes[idx].feature] < _nodes[idx].v ? 1 : _nodes[idx].idx;
      }
      sum += _nodes[idx].v;
      idx += _nodes[idx].idx;
    }

    return sum;
  }

  const std::vector<int>& sizes() const;

  int total_nodes() const;

  std::vector<int> feature_frequency() const;

 private:
  std::vector<FastNode> _nodes;

  std::vector<int> _sizes;
};

} // namespace ml
