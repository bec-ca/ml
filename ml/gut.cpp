#include "gut.hpp"

#include "ml/fast_tree.hpp"
#include "ml/gut_config.hpp"
#include "yasf/serializer.hpp"
#include "yasf/value.hpp"

using std::make_shared;
using std::make_unique;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;
using yasf::Value;

namespace ml {

namespace {

template <class T> T square(T v) { return v * v; }

double compute_loss(LossFunction lf, double residual)
{
  switch (lf) {
  case LossFunction::L1:
    return std::abs(residual);
  case LossFunction::L2:
    return square(residual);
  }
}

double compute_grad(LossFunction lf, double residual)
{
  switch (lf) {
  case LossFunction::L1:
    return residual < 0 ? -1.0 : 1.0;
  case LossFunction::L2:
    return residual;
  }
}

struct GutImpl : public Gut {
 public:
  using ptr = shared_ptr<GutImpl>;

  GutImpl(const GutConfig& config) : _config(config)
  {
    add_tree();
    update_fast_trees();
  }

  GutImpl(const GutConfig& config, vector<Node>&& trees)
      : _config(config), _trees(std::move(trees))
  {
    update_fast_trees();
  }

  virtual double run_step(const std::vector<DataPoint>& batch) override
  {
    double loss_sum = 0;
    for (auto& dp : batch) {
      assert(std::ssize(dp.features) == _config.num_features);
      double residual = dp.label;
      for (auto& t : _trees) { residual -= t.predict(dp.features); }
      loss_sum += compute_loss(_config.loss_function, residual);
      _trees.back().add_leaf_stats(dp.features, residual);
      double grad =
        compute_grad(_config.loss_function, residual) / _trees.size();
      for (auto& t : _trees) { t.add_grad(dp.features, grad); }
    }

    bool should_add_tree = std::ssize(_trees) < _config.max_num_trees;
    for (auto& p : _trees) {
      p.apply_grad(_config.learning_rate, _config);
      if (p.nodes() < _config.max_tree_nodes) {
        p.maybe_split(_config.max_tree_height, _config);
        should_add_tree = false;
      }
    }

    if (should_add_tree) { add_tree(); }

    update_fast_trees();

    return loss_sum / batch.size();
  }

  virtual const FastTree& fast_trees() const override { return *_fast_trees; }

  virtual Value::ptr to_yasf_value() const override
  {
    return yasf::ser(_trees);
  }

  virtual const GutConfig& config() const override { return _config; }

  static bee::OrError<ptr> of_yasf_value(
    const Value::ptr& value, const GutConfig& config)
  {
    bail(trees, yasf::des<vector<Node>>(value, config));
    return make_shared<GutImpl>(config, std::move(trees));
  }

  virtual int num_trees() const override { return _trees.size(); }

  virtual void update_from(const Gut& other, double lambda) override
  {
    auto& gut_impl = dynamic_cast<const GutImpl&>(other);
    _update_from(gut_impl, lambda);
  }

 private:
  void _update_from(const GutImpl& other, double lambda)
  {
    while (_trees.size() < other._trees.size()) { add_tree(); }
    for (int i = 0; i < std::ssize(_trees); ++i) {
      _trees.at(i).update_from(other._trees.at(i), lambda, _config);
    }
    update_fast_trees();
  }

  void add_tree() { _trees.push_back(ml::Node::create_leaf(_config)); }

  void update_fast_trees() { _fast_trees = make_unique<FastTree>(_trees); }

  const GutConfig _config;
  vector<Node> _trees;
  unique_ptr<FastTree> _fast_trees;
};

} // namespace

Gut::~Gut() {}

Gut::ptr Gut::create(const GutConfig& config)
{
  return make_shared<GutImpl>(config);
}

bee::OrError<Gut::ptr> Gut::of_yasf_value(
  const Value::ptr& value, const GutConfig& config)
{
  return GutImpl::of_yasf_value(value, config);
}

} // namespace ml
