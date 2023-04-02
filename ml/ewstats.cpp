#include "ewstats.hpp"

#include <algorithm>
#include <cmath>

namespace ml {

Ewstats::Ewstats(double lambda)
    : _lambda(lambda), _sum(0), _sum_sq(0), _weight(0)
{}

void Ewstats::add(double value)
{
  _sum = _sum * _lambda + value;
  _sum_sq = _sum_sq * _lambda + value * value;
  _weight = _weight * _lambda + 1;
}

double Ewstats::avg() const { return _sum / _weight; }

double Ewstats::var() const
{
  return std::max(0.0, (_sum_sq - (_sum * _sum) / _weight) / _weight);
}

double Ewstats::stddev() const { return sqrt(var()); }

} // namespace ml
