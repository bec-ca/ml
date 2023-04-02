#include "ewcovar.hpp"

namespace ml {

Ewcovar::Ewcovar(double lambda)
    : _lambda(lambda), _sum1(0), _sum2(0), _prod_sum(0), _weight(0)
{}

void Ewcovar::add(double v1, double v2)
{
  _sum1 = _sum1 * _lambda + v1;
  _sum2 = _sum2 * _lambda + v2;

  _prod_sum = _prod_sum * _lambda + v1 * v2;
  _weight = _weight * _lambda + 1.0;
}

double Ewcovar::covar() const
{
  return (_prod_sum - _sum1 * _sum2 / _weight) / _weight;
}

} // namespace ml
