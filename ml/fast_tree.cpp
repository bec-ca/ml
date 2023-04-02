#include "fast_tree.hpp"
#include <functional>

using std::function;
using std::vector;

namespace ml {

namespace {

vector<FastNode> make_nodes(const vector<Node>& trees)
{
  vector<FastNode> out;

  function<int(const Node& node)> create_rec;

  int next_tree_idx = 0;

  create_rec = [&](const Node& node) -> int32_t {
    int32_t idx = out.size();
    if (node.is_leaf()) {
      out.push_back(
        FastNode::make_leaf(node.leaf_value(), next_tree_idx - idx));
    } else if (node.is_inner()) {
      out.push_back(
        FastNode::make_inner(node.inner_feature(), node.inner_threshold()));
      create_rec(node.less());
      out[idx].idx = create_rec(node.greater()) - idx;
    } else {
      assert(false);
    }
    return idx;
  };

  for (const auto& t : trees) {
    next_tree_idx += t.nodes();
    create_rec(t);
    assert(std::ssize(out) == next_tree_idx);
  }

  return out;
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
// FastNode
//

FastNode FastNode::make_leaf(float v, int32_t next_tree)
{
  return FastNode{
    .v = v,
    .feature = -1,
    .idx = next_tree,
  };
}

FastNode FastNode::make_inner(int32_t feature, float threshold)
{
  return FastNode{
    .v = threshold,
    .feature = feature,
    .idx = -1,
  };
}

////////////////////////////////////////////////////////////////////////////////
// FastTree
//

FastTree::~FastTree() {}

FastTree::FastTree(const FastTree&) = default;
FastTree::FastTree(FastTree&&) = default;

FastTree::FastTree(const vector<Node>& trees) : _nodes(make_nodes(trees))
{
  for (auto& n : trees) { _sizes.push_back(n.nodes()); }
}

const vector<int>& FastTree::sizes() const { return _sizes; }

int FastTree::total_nodes() const { return _nodes.size(); }

vector<int> FastTree::feature_frequency() const
{
  vector<int> output;
  for (auto& n : _nodes) {
    if (n.feature == -1) { continue; }
    output.resize(std::max<int>(std::ssize(output), n.feature + 1), 0);
    output.at(n.feature)++;
  }
  return output;
}

} // namespace ml
