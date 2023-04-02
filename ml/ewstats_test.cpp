#include "ewstats.hpp"

#include "bee/format.hpp"
#include "bee/testing.hpp"

using bee::print_line;

namespace ml {
namespace {

TEST(basic)
{
  auto run = [](double lambda) {
    print_line("-------------------");
    print_line("lambda: $", lambda);
    Ewstats stats(lambda);
    for (int i = 0; i < 1000; i++) { stats.add(i % 3); }
    print_line(stats.avg());
    print_line(stats.var());
    print_line(stats.stddev());
  };
  run(0.9);
  run(0.99);
  run(0.999);
  run(0.9999);
  run(1.0);
}

TEST(decay)
{
  auto run = [](double lambda, int steps) {
    print_line("-------------------");
    print_line("lambda: $, steps: $", lambda, steps);
    Ewstats stats(lambda);
    stats.add(1000);
    for (int i = 0; i < steps; i++) { stats.add(1); }
    print_line("avg: $", stats.avg());
    print_line("var: $", stats.var());
    print_line("stdev: $", stats.stddev());
  };
  run(0.999, 1);
  run(0.999, 10);
  run(0.999, 100);
  run(0.999, 1000);
  run(0.999, 10000);
  run(0.999, 100000);

  run(0.99, 1);
  run(0.99, 10);
  run(0.99, 100);
  run(0.99, 1000);
  run(0.99, 10000);
  run(0.99, 100000);
}

} // namespace
} // namespace ml
