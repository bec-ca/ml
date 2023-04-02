#include "rng.hpp"
#include <cstdint>
#include <random>

using std::make_unique;
using std::unique_ptr;

namespace ml {

namespace {

struct Mersenne : public Rng {
 public:
  Mersenne(uint32_t seed) : _rng(seed) {}

  virtual ~Mersenne() {}

  virtual uint32_t operator()() override { return _rng(); }

 private:
  std::mt19937 _rng;
};

static_assert(Rng::min() == std::mt19937::min());
static_assert(Rng::max() == std::mt19937::max());

} // namespace

////////////////////////////////////////////////////////////////////////////////
// Rng
//

Rng::~Rng() {}

unique_ptr<Rng> Rng::create(uint32_t seed)
{
  return make_unique<Mersenne>(seed);
}

} // namespace ml
