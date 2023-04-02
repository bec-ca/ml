#pragma once

namespace ml {

struct Ewcovar {
 public:
  Ewcovar(double lambda);

  void add(double v1, double v2);

  double covar() const;

 private:
  const double _lambda;

  double _sum1;
  double _sum2;
  double _prod_sum;
  double _weight;
};

} // namespace ml
