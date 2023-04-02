#include "ewcovar.hpp"

#include "ewstats.hpp"

#include "bee/format.hpp"
#include "bee/testing.hpp"

#include <random>

using bee::print_line;

namespace ml {
namespace {

TEST(basic)
{
  auto run = [](double lambda) {
    print_line("--------------------");
    print_line("lambda: $", lambda);
    Ewcovar covar(lambda);
    Ewstats stats1(lambda);
    Ewstats stats2(lambda);
    for (int i = 0; i < 1000; i++) {
      int v1 = i % 3;
      int v2 = v1 + 2;
      covar.add(v1, v2);
      stats1.add(v1);
      stats2.add(v2);
    }
    print_line(
      "covar:$ corr:$",
      covar.covar(),
      covar.covar() / stats1.stddev() / stats2.stddev());
  };

  run(0.9);
  run(0.99);
  run(0.999);
  run(0.9999);
  run(1);
}

TEST(inverse)
{
  auto run = [](double lambda) {
    print_line("--------------------");
    print_line("lambda: $", lambda);
    Ewcovar covar(lambda);
    Ewstats stats1(lambda);
    Ewstats stats2(lambda);
    for (int i = 0; i < 1000; i++) {
      int v1 = i % 3;
      int v2 = -v1 + 2;
      covar.add(v1, v2);
      stats1.add(v1);
      stats2.add(v2);
    }
    print_line(
      "covar:$ corr:$",
      covar.covar(),
      covar.covar() / stats1.stddev() / stats2.stddev());
  };

  run(0.9);
  run(0.99);
  run(0.999);
  run(0.9999);
  run(1);
}

TEST(randomized)
{
  auto run = [](double lambda) {
    print_line("--------------------");
    print_line("lambda: $", lambda);
    std::mt19937 rng(0);
    std::uniform_real_distribution<double> dist(0, 1);
    Ewcovar covar(lambda);
    Ewstats stats1(lambda);
    Ewstats stats2(lambda);
    for (int i = 0; i < 1000; i++) {
      double v1 = dist(rng) * 10;
      double v2 = v1 + dist(rng) * 10;
      covar.add(v1, v2);
      stats1.add(v1);
      stats2.add(v2);
    }
    print_line(
      "covar:$ corr:$",
      covar.covar(),
      covar.covar() / stats1.stddev() / stats2.stddev());
  };

  run(0.9);
  run(0.99);
  run(0.999);
  run(0.9999);
  run(1);
}

} // namespace
} // namespace ml
