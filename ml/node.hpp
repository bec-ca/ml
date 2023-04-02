#pragma once

#include "datapoint.hpp"
#include "gut_config.hpp"
#include "rng.hpp"

#include "yasf/value.hpp"

#include <memory>
#include <type_traits>
#include <variant>

namespace ml {

struct Node;
struct InnerNode;
struct LeafNode;

struct Node {
 public:
  using value_type =
    std::variant<std::unique_ptr<LeafNode>, std::unique_ptr<InnerNode>>;

  Node(std::unique_ptr<LeafNode>&& leaf);
  Node(std::unique_ptr<InnerNode>&& leaf);

  Node(const Node& other) = delete;
  Node(Node&& other);

  Node& operator=(const Node& other) = delete;
  Node& operator=(Node&& other);

  ~Node();

  template <class F>
  inline std::invoke_result_t<F, const std::unique_ptr<LeafNode>&> visit(
    F&& f) const
  {
    return std::visit(std::forward<F>(f), _var);
  }

  template <class F>
  inline std::invoke_result_t<F, const std::unique_ptr<LeafNode>&> visit(F&& f)
  {
    return std::visit(std::forward<F>(f), _var);
  }

  double predict(const std::vector<float>& dp) const;

  void add_leaf_stats(const std::vector<float>& features, double residual);

  void add_grad(const std::vector<float>& features, double grad);

  double max_abs_value() const;
  void apply_grad(double learning_rate, const GutConfig& config);
  std::string to_string() const;
  void maybe_prune();
  int nodes() const;

  void maybe_split(int max_depth, const GutConfig& config);

  static Node create_leaf(const GutConfig& config);

  yasf::Value::ptr to_yasf_value() const;
  static bee::OrError<Node> of_yasf_value(
    const yasf::Value::ptr& value, const GutConfig& config);

  // Check node type
  bool is_leaf() const;
  bool is_inner() const;

  // Leaf node accessors
  double leaf_value() const;

  // Inner node accessors
  double inner_threshold() const;
  int inner_feature() const;
  const Node& less() const;
  const Node& greater() const;

  void update_from(const Node& other, double lambda, const GutConfig&);

 private:
  value_type _var;
};

} // namespace ml
