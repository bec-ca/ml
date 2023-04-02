#pragma once

namespace ml {

struct Ewstats {
 public:
  Ewstats(double lambda);

  void add(double value);

  double avg() const;
  double var() const;
  double stddev() const;

 private:
  const double _lambda;

  double _sum;
  double _sum_sq;
  double _weight;
};

} // namespace ml
