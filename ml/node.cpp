#include "node.hpp"

#include "ewcovar.hpp"
#include "ewma.hpp"
#include "ewstats.hpp"
#include "gut_config.hpp"

#include "bee/format.hpp"
#include "bee/nref.hpp"
#include "yasf/serializer.hpp"
#include "yasf/value.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <type_traits>

using bee::nref;
using std::make_unique;
using std::string;
using std::unique_ptr;
using std::vector;

namespace ml {

namespace {

struct Updatable {
 public:
  Updatable(double value) : _value(value), _grad_sum(0), _count(0) {}

  void add_grad(double grad)
  {
    _grad_sum += grad;
    _count++;
  }

  void apply(double learning_rate)
  {
    if (_count > 0) {
      _value += _grad_sum / _count * learning_rate;
      _grad_sum = 0;
      _count = 0;
    }
  }

  void update_from(const Updatable& value, double lambda)
  {
    _value += lambda * (value._value - _value);
  }

  double value() const { return _value; }

  string to_string() const { return bee::format(value()); }

  int count() const { return _count; }

 private:
  double _value;
  double _grad_sum;
  int _count;
};

} // namespace

////////////////////////////////////////////////////////////////////////////////
// LeafNode
//

struct LeafNode {
 public:
  LeafNode(double value, const GutConfig& config)
      : _value(value), _num_features(config.num_features)
  {
    _feature_stats.reserve(config.num_features);
    for (int i = 0; i < config.num_features; i++) {
      _feature_stats.emplace_back(config.ew_lambda);
      _feature_covar.emplace_back(config.ew_lambda);
    }
  }

  void add_leaf_stats(const vector<float>& features, double label)
  {
    for (int i = 0; i < _num_features; i++) {
      _feature_stats[i].add(features[i]);
      _feature_covar[i].add(features[i], label);
    }
  }

  void add_grad(double grad) { _value.add_grad(grad); }

  void apply_grad(double learning_rate)
  {
    _total_samples_seen += _value.count();
    _value.apply(learning_rate);
  }

  void update_from(const LeafNode& other, double lambda)
  {
    _value.update_from(other._value, lambda);
  }

  inline double max_abs_value() const { return std::abs(value()); }

  double value() const { return _value.value(); }

  string to_string() const { return bee::format("[v:$]", _value); }

  unique_ptr<InnerNode> maybe_split(const GutConfig& config)
  {
    if (_total_samples_seen < config.min_samples_to_split) { return nullptr; }

    int selected_feature = -1;
    double largest_corr = -1e10;
    for (int i = 0; i < _num_features; i++) {
      double stddev = _feature_stats[i].stddev();
      if (std::abs(stddev) <= 1e-9) { continue; }

      double abs_corr = std::abs(_feature_covar[i].covar() / stddev);
      if (selected_feature == -1 || abs_corr > largest_corr) {
        selected_feature = i;
        largest_corr = abs_corr;
      }
    }

    if (selected_feature == -1) { return nullptr; }

    auto threshold = _feature_stats[selected_feature].avg();

    return make_unique<InnerNode>(
      selected_feature,
      threshold,
      make_unique<LeafNode>(_value.value(), config),
      make_unique<LeafNode>(_value.value(), config));
  }

  int nodes() const { return 1; }

  using fmt_type = double;

  yasf::Value::ptr to_yasf_value() const
  {
    return yasf::ser(fmt_type(_value.value()));
  }

  static bee::OrError<unique_ptr<LeafNode>> of_yasf_value(
    const yasf::Value::ptr& v, const GutConfig& config)
  {
    using std::get;
    bail(t, yasf::des<fmt_type>(v));
    return make_unique<LeafNode>(t, config);
  }

 private:
  Updatable _value;

  const int _num_features;

  int _total_samples_seen = 0;

  vector<Ewstats> _feature_stats;
  vector<Ewcovar> _feature_covar;
};

////////////////////////////////////////////////////////////////////////////////
// InnerNode
//

struct InnerNode {
 public:
  InnerNode(int feature, double threshold, Node&& less, Node&& greater)
      : _feature(feature),
        _threshold(threshold),
        _less(std::move(less)),
        _greater(std::move(greater))
  {}

  inline Node& get_child(const vector<float>& f)
  {
    return is_less(f) ? _less : _greater;
  }

  inline const Node& get_child(const vector<float>& f) const
  {
    return is_less(f) ? _less : _greater;
  }

  inline double predict(const vector<float>& f) const
  {
    return get_child(f).predict(f);
  }

  inline bool is_less(const vector<float>& f) const
  {
    return f[_feature] < _threshold;
  }

  void add_grad(const vector<float>& features, double grad)
  {
    get_child(features).add_grad(features, grad);

    auto dist = std::abs(features[_feature] - _threshold);
    auto threshold_grad = std::min(dist, std::abs(grad * dist));

    if (is_less(features)) {
      _threshold_grad -= threshold_grad;
    } else {
      _threshold_grad += threshold_grad;
    }

    _count++;
  }

  void add_leaf_stats(const vector<float>& features, double residual)
  {
    get_child(features).add_leaf_stats(features, residual);
  }

  void apply_grad(double learning_rate, const GutConfig& config)
  {
    if (_count > 0) {
      if (config.update_threshold) {
        _threshold += _threshold_grad * learning_rate / _count;
      }
      _threshold_grad = 0.0;
      _count = 0;

      learning_rate *= config.lr_decay;

      _less.apply_grad(learning_rate, config);
      _greater.apply_grad(learning_rate, config);

      _grad_updates++;
    }
  }

  void update_from(
    const InnerNode& other, double lambda, const GutConfig& config)
  {
    assert(_feature == other._feature);
    _threshold += lambda * (other._threshold - _threshold);
    _less.update_from(other._less, lambda, config);
    _greater.update_from(other._greater, lambda, config);
  }

  string to_string() const
  {
    return bee::format(
      "[t:$ f:$ l:$ g:$]", _threshold, _feature, _less, _greater);
  }

  double max_abs_value() const
  {
    return std::max(_less.max_abs_value(), _greater.max_abs_value());
  }

  nref<Node> maybe_prune()
  {
    return nullptr;
    if (_grad_updates < 10) { return nullptr; }
    const double prune_threshold = 0.3;
    if (_less.max_abs_value() < prune_threshold) {
      return _greater;
    } else if (_greater.max_abs_value() < prune_threshold) {
      return _less;
    } else {
      _less.maybe_prune();
      _greater.maybe_prune();
      return nullptr;
    }
  }

  void maybe_split(int max_depth, const GutConfig& config)
  {
    _less.maybe_split(max_depth, config);
    _greater.maybe_split(max_depth, config);
  }

  int nodes() const { return 1 + _less.nodes() + _greater.nodes(); }

  using fmt_type = std::tuple<int, double, yasf::Value::ptr, yasf::Value::ptr>;

  yasf::Value::ptr to_yasf_value() const
  {
    return yasf::ser(fmt_type(
      _feature, _threshold, _less.to_yasf_value(), _greater.to_yasf_value()));
  }

  static bee::OrError<unique_ptr<InnerNode>> of_yasf_value(
    const yasf::Value::ptr& v, const GutConfig& config)
  {
    bail(t, yasf::des<fmt_type>(v));
    using std::get;
    bail(left, Node::of_yasf_value(get<2>(t), config));
    bail(right, Node::of_yasf_value(get<3>(t), config));
    return make_unique<InnerNode>(
      get<0>(t), get<1>(t), std::move(left), std::move(right));
  }

  double threshold() const { return _threshold; }

  int feature() const { return _feature; }

  const Node& less() const { return _less; }

  const Node& greater() const { return _greater; }

 private:
  const int _feature;
  double _threshold;
  Node _less;
  Node _greater;
  double _threshold_grad = 0.0;
  int _count = 0;
  int _grad_updates = 0;
};

////////////////////////////////////////////////////////////////////////////////
// Node
//

Node::Node(std::unique_ptr<LeafNode>&& leaf) : _var(std::move(leaf)) {}
Node::Node(std::unique_ptr<InnerNode>&& leaf) : _var(std::move(leaf)) {}

Node::Node(Node&& other) = default;

Node& Node::operator=(Node&& other) = default;

Node::~Node() {}

template <class T>
constexpr bool is_inner_node =
  std::is_same_v<std::decay_t<T>, unique_ptr<InnerNode>>;

template <class T>
constexpr bool is_leaf_node =
  std::is_same_v<std::decay_t<T>, unique_ptr<LeafNode>>;

double Node::predict(const vector<float>& dp) const
{
  return visit([&]<class T>(T&& c) {
    if constexpr (is_inner_node<T>) {
      return c->predict(dp);
    } else {
      return c->value();
    }
  });
}

void Node::add_grad(const vector<float>& features, double grad)
{
  visit([&]<class T>(T&& c) {
    if constexpr (is_inner_node<T>) {
      c->add_grad(features, grad);
    } else {
      c->add_grad(grad);
    }
  });
}

void Node::add_leaf_stats(const vector<float>& features, double residual)
{
  visit([&](auto&& c) { c->add_leaf_stats(features, residual); });
}

void Node::apply_grad(double learning_rate, const GutConfig& config)
{
  visit([&]<class T>(T& c) {
    if constexpr (is_inner_node<T>) {
      c->apply_grad(learning_rate, config);
    } else {
      c->apply_grad(learning_rate);
    }
  });
}

void Node::update_from(
  const Node& other, double lambda, const GutConfig& config)
{
  auto new_node = visit([&]<class T>(T& t) -> unique_ptr<InnerNode> {
    return other.visit([&]<class U>(const U& u) -> unique_ptr<InnerNode> {
      if constexpr (is_inner_node<T>) {
        if constexpr (is_inner_node<U>) {
          t->update_from(*u, lambda, config);
          return nullptr;
        } else {
          assert(false);
        }
      } else {
        if constexpr (is_inner_node<U>) {
          auto inner = make_unique<InnerNode>(
            u->feature(),
            u->threshold(),
            make_unique<LeafNode>(t->value(), config),
            make_unique<LeafNode>(t->value(), config));
          inner->update_from(*u, lambda, config);
          return inner;
        } else {
          t->update_from(*u, lambda);
          return nullptr;
        }
      }
    });
  });
  if (new_node != nullptr) { _var = std::move(new_node); }
}

double Node::max_abs_value() const
{
  return visit([&](const auto& c) { return c->max_abs_value(); });
}

int Node::nodes() const
{
  return visit([](auto&& c) { return c->nodes(); });
}

string Node::to_string() const
{
  return visit([&]<class T>(const T& c) { return c->to_string(); });
}

void Node::maybe_prune()
{
  visit([&]<class T>(const T& c) {
    if constexpr (is_inner_node<T>) {
      auto p = c->maybe_prune();
      if (p != nullptr) {
        auto c = std::move(*p);
        *this = std::move(c);
      }
    }
  });
}

void Node::maybe_split(int max_depth, const GutConfig& config)
{
  if (max_depth <= 0) { return; }
  visit([&]<class T>(const T& c) {
    if constexpr (is_inner_node<T>) {
      c->maybe_split(max_depth - 1, config);
    } else {
      auto p = c->maybe_split(config);
      if (p != nullptr) { _var = std::move(p); }
    }
  });
}

Node Node::create_leaf(const GutConfig& config)
{
  return make_unique<LeafNode>(0, config);
}

yasf::Value::ptr Node::to_yasf_value() const
{
  using p = std::pair<string, yasf::Value::ptr>;
  return visit([&]<class T>(const T& c) {
    if constexpr (is_inner_node<T>) {
      return yasf::ser(p("InnerNode", c->to_yasf_value()));
    } else {
      return yasf::ser(p("LeafNode", c->to_yasf_value()));
    }
  });
}

bee::OrError<Node> Node::of_yasf_value(
  const yasf::Value::ptr& value, const GutConfig& config)
{
  using p = std::pair<string, yasf::Value::ptr>;
  bail(v, yasf::des<p>(value));
  if (v.first == "InnerNode") {
    return InnerNode::of_yasf_value(v.second, config);
  } else if (v.first == "LeafNode") {
    return LeafNode::of_yasf_value(v.second, config);
  } else {
    return bee::Error("Invalid file");
  }
}

bool Node::is_leaf() const
{
  return visit([&]<class T>(const T&) { return !is_inner_node<T>; });
}

bool Node::is_inner() const
{
  return visit([&]<class T>(const T&) { return is_inner_node<T>; });
}

double Node::leaf_value() const
{
  return visit([&]<class T>(const T& c) -> double {
    if constexpr (is_inner_node<T>) {
      assert(false);
    } else {
      return c->value();
    }
  });
}

double Node::inner_threshold() const
{
  return visit([&]<class T>(const T& c) -> double {
    if constexpr (is_inner_node<T>) {
      return c->threshold();
    } else {
      assert(false);
    }
  });
}

int Node::inner_feature() const
{
  return visit([&]<class T>(const T& c) -> int {
    if constexpr (is_inner_node<T>) {
      return c->feature();
    } else {
      assert(false);
    }
  });
}

const Node& Node::less() const
{
  return visit([&]<class T>(const T& c) -> const Node& {
    if constexpr (is_inner_node<T>) {
      return c->less();
    } else {
      assert(false);
    }
  });
}

const Node& Node::greater() const
{
  return visit([&]<class T>(const T& c) -> const Node& {
    if constexpr (is_inner_node<T>) {
      return c->greater();
    } else {
      assert(false);
    }
  });
}

} // namespace ml
