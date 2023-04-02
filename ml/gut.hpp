#pragma once

#include "ml/datapoint.hpp"
#include "ml/fast_tree.hpp"
#include "ml/gut_config.hpp"

#include "yasf/value.hpp"

#include <memory>
#include <optional>
#include <vector>

namespace ml {

struct Gut {
 public:
  using ptr = std::shared_ptr<Gut>;

  virtual ~Gut();

  static ptr create(const GutConfig& config);

  static bee::OrError<ptr> of_yasf_value(
    const yasf::Value::ptr&, const GutConfig& config);
  virtual yasf::Value::ptr to_yasf_value() const = 0;

  virtual double run_step(const std::vector<DataPoint>& batch) = 0;

  virtual const FastTree& fast_trees() const = 0;

  virtual const GutConfig& config() const = 0;

  virtual void update_from(const Gut& other, double lambda) = 0;

  virtual int num_trees() const = 0;
};

} // namespace ml
