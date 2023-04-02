#pragma once

#include "yasf/value.hpp"

namespace ml {

struct Ewma {
 public:
  explicit Ewma(double lambda);

  void add(double value);

  double avg() const;

  yasf::Value::ptr to_yasf_value() const;
  static bee::OrError<Ewma> of_yasf_value(const yasf::Value::ptr& value);

 private:
  Ewma(double lambda, double sum, double weight);

  const double _lambda;
  double _sum;
  double _weight;
};

} // namespace ml
