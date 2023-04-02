#include "ewma.hpp"

#include "yasf/serializer.hpp"

namespace ml {

Ewma::Ewma(double lambda, double sum, double weight)
    : _lambda(lambda), _sum(sum), _weight(weight)
{}

Ewma::Ewma(double lambda) : Ewma(lambda, 0, 0) {}

void Ewma::add(double value)
{
  _sum = _sum * _lambda + value;
  _weight = _weight * _lambda + 1;
}

double Ewma::avg() const { return _sum / _weight; }

using fmt_type = std::tuple<double, double, double>;

yasf::Value::ptr Ewma::to_yasf_value() const
{
  return yasf::ser(fmt_type(_lambda, _sum, _weight));
}

bee::OrError<Ewma> Ewma::of_yasf_value(const yasf::Value::ptr& v)
{
  using std::get;
  bail(t, yasf::des<fmt_type>(v));
  return Ewma(get<0>(t), get<1>(t), get<2>(t));
}

} // namespace ml
