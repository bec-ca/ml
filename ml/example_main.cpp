#include "datapoint.hpp"
#include "fast_tree.hpp"
#include "ml/gut_config.hpp"
#include "node.hpp"

#include "bee/file_writer.hpp"
#include "bee/format.hpp"
#include "bee/format_memory.hpp"
#include "bee/format_vector.hpp"
#include "bee/time.hpp"
#include "command/command_builder.hpp"
#include "command/group_builder.hpp"
#include "csv/csv_file.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <utility>

using bee::print_line;
using bee::Time;
using std::string;
using std::vector;

namespace ml {
namespace {

string zero_pad(int n, int d)
{
  auto s = bee::format(n);
  while (std::ssize(s) < d) { s = "0" + s; }
  return s;
}

DataPoint parse_row(const csv::CsvRow& row)
{
  vector<float> features;
  std::optional<int> label = -1;
  for (const auto& value : row) {
    if (value.col() == "target") {
      label = stoi(value.value());
    } else if (value.col() == "ID_code") {
      continue;
    } else {
      features.push_back(std::stod(value.value()));
    }
  }
  assert(label.has_value() && (*label == 0 || *label == 1));
  return DataPoint{
    .features = std::move(features),
    .label = *label == 1 ? 1.0 : -1.0,
  };
}

bee::OrError<vector<DataPoint>> read_csv(const string& filename)
{
  bail(csv, csv::CsvFile::open_csv(filename));
  size_t num_features = csv->header().size() - 2;
  vector<DataPoint> samples;
  print_line("Reading training data...");
  while (true) {
    must(row, csv->next_row());
    if (!row.has_value()) { break; }
    auto sample = parse_row(*row);
    assert(sample.size() == num_features);
    samples.push_back(std::move(sample));
  }
  return samples;
}

struct DatasetRange {
 public:
  using container = vector<DataPoint>;

  DatasetRange(container::const_iterator begin, container::const_iterator end)
      : _begin(begin), _end(end)
  {}

  inline auto begin() const { return _begin; }
  inline auto end() const { return _end; }

  auto size() const { return _end - _begin; }

 private:
  container::const_iterator _begin;
  container::const_iterator _end;
};

constexpr static double tree_learning_rate = 0.1;

struct Predictors {
 public:
  Predictors(const GutConfig& config) : _config(config) { add_predictor(); }

  vector<int> num_nodes() const
  {
    vector<int> output;
    for (auto& p : _predictors) { output.push_back(p.nodes()); }
    return output;
  }

  int total_nodes() const
  {
    int output = 0;
    for (auto& p : _predictors) { output += p.nodes(); }
    return output;
  }

  double pred_one(const vector<float>& f) const
  {
    double sum = 0.0;
    for (auto& p : _predictors) { sum += p.predict(f) * tree_learning_rate; }
    return sum;
  }

  void train_step(const DatasetRange& dataset, double learning_rate)
  {
    for (auto& dp : dataset) {
      double residual = dp.label;
      int num_predictors = std::ssize(_predictors);
      for (int i = 0; i < num_predictors - 1; i++) {
        auto& p = _predictors[i];
        residual -= p.predict(dp.features) * tree_learning_rate;
      }
      _predictors.back().add_grad(dp.features, residual);
    }
    _predictors.back().apply_grad(learning_rate, _config);
  }

  void prune_and_split()
  {
    for (auto& p : _predictors) {
      p.maybe_prune();
      if (p.nodes() < _config.max_tree_nodes) {
        p.maybe_split(_config.max_tree_height, _config);
      }
    }
  }

  void maybe_split()
  {
    bool should_create_tree = true;
    for (auto& p : _predictors) {
      if (p.nodes() < _config.max_tree_nodes) {
        p.maybe_split(_config.max_tree_height, _config);
        should_create_tree = false;
      }
    }
    if (should_create_tree) { add_predictor(); }
  }

  double accuracy(const vector<DataPoint>& dataset)
  {
    int correct = 0;
    for (auto& dp : dataset) {
      bool pred_bool = pred_one(dp.features) >= 0.0;
      if (pred_bool == dp.label) { correct++; }
    }
    return double(correct) / double(dataset.size());
  };

  double loss(const vector<DataPoint>& dataset)
  {
    double loss_sum = 0;
    for (auto& dp : dataset) {
      double pred = pred_one(dp.features);
      double d = pred - dp.label;
      loss_sum += d * d;
    }
    return loss_sum / dataset.size();
  };

  void train_epoch(
    const vector<DataPoint>& dataset, size_t num_chunks, double learning_rate)
  {
    for (size_t k = 0; k < num_chunks; k++) {
      auto s = dataset.size();
      int begin = (k * s) / num_chunks;
      int end = ((k + 1) * s) / num_chunks;

      auto b = dataset.begin();
      train_step(DatasetRange(b + begin, b + end), learning_rate);
    }
  }

  string to_string() const { return bee::format(_predictors); }

  FastTree create_fast_trees() const
  {
    FastTree out(_predictors);
    return out;
  }

 private:
  void add_predictor() { _predictors.push_back(Node::create_leaf(_config)); }

  const GutConfig _config;
  vector<Node> _predictors;
};

bee::OrError<bee::Unit> train_kagle_main(const string& training_filename)
{
  const int dim = 20;
  const int tree_depth = 10;
  const int iters = 800;
  const int num_chunks = 200;
  const int max_tree_nodes = 1000;
  const double learning_rate = 0.1;

  std::random_device rd;
  auto rng = Rng::create(rd());

  bail(training_dataset, read_csv(training_filename));
  print_line("Num data points: $", training_dataset.size());

  std::shuffle(training_dataset.begin(), training_dataset.end(), *rng);

  vector<DataPoint> testing_dataset(
    training_dataset.end() - training_dataset.size() / 10,
    training_dataset.end());
  training_dataset.resize(training_dataset.size() - testing_dataset.size());

  {
    vector<DataPoint> augmented_dataset;
    for (auto& dp : training_dataset) {
      if (dp.label) {
        for (int j = 0; j < 10; j++) { augmented_dataset.push_back(dp); }
      } else {
        augmented_dataset.push_back(dp);
      }
    }
    training_dataset = augmented_dataset;
    std::shuffle(training_dataset.begin(), training_dataset.end(), *rng);
  }

  const int num_features = training_dataset[0].size();
  print_line("Num features: $", num_features);

  std::uniform_real_distribution<double> dist(-dim, dim);

  Predictors predictors({
    .num_features = num_features,
    .max_tree_nodes = max_tree_nodes,
    .max_tree_height = tree_depth,
    .lr_decay = 1.0,
  });

  auto roc = [&](const vector<DataPoint>& dataset) {
    vector<std::pair<double, bool>> evals;
    int num_positives = 0;
    for (auto& dp : dataset) {
      auto p = predictors.pred_one(dp.features);
      evals.emplace_back(p, dp.label);
      if (dp.label) { num_positives++; }
    }
    int num_negatives = dataset.size() - num_positives;
    std::stable_sort(evals.begin(), evals.end(), [](auto&& p1, auto&& p2) {
      return p1.first < p2.first;
    });

    int y = 0;
    double area = 0;
    for (int i = 0; i < std::ssize(evals); i++) {
      if (evals[i].second) {
        area += y;
      } else {
        y++;
      }
    }
    return area / double(num_positives) / double(num_negatives);
  };

  for (int i = 0; i < iters; i++) {
    print_line("-------------------------------------");
    predictors.maybe_split();

    print_line("Nodes: $", predictors.num_nodes());
    print_line("Total nodes: $", predictors.total_nodes());
    print_line("Step $", i);
    print_line("Test accuracy: $%", predictors.accuracy(testing_dataset) * 100);
    print_line("Test roc: $", roc(testing_dataset));
    print_line(
      "Training accuracy: $%", predictors.accuracy(training_dataset) * 100);
    print_line("Training roc: $", roc(training_dataset));
    auto start = Time::monotonic();
    for (int j = 0; j < 10; j++) {
      predictors.train_epoch(training_dataset, num_chunks, learning_rate);
      std::shuffle(training_dataset.begin(), training_dataset.end(), *rng);
    }
    auto ending = Time::monotonic();
    print_line("Took: $", ending - start);
  }
  print_line("Test accuracy: $%", predictors.accuracy(testing_dataset) * 100);
  print_line("Test roc: $", roc(testing_dataset));
  print_line(
    "Training accuracy: $%", predictors.accuracy(training_dataset) * 100);
  print_line("Training roc: $", roc(training_dataset));

  return bee::ok();
}

bee::OrError<bee::Unit> train_image_main()
{
  std::random_device rd;
  auto rng = Rng::create(rd());

  const int dim = 500;
  const int num_features = 2;
  const int tree_depth = 200;
  const int iters = 200;
  const int test_dataset_size = 1000000;
  const int num_chunks = 10;
  const int max_tree_nodes = 1 << 10;
  const int training_batch_size = 1000000;
  const double learning_rate = 1.0;

  print_line("Num features: $", num_features);

  std::uniform_real_distribution<double> dist(-dim, dim);

  auto make_dataset = [&](int size) {
    vector<DataPoint> dataset;

    for (int i = 0; i < size; i++) {
      vector<float> features;
      for (int j = 0; j < num_features; j++) { features.push_back(dist(*rng)); }

      double d = 1.0;
      for (auto& v : features) { d *= cos(v / 50); }

      double label = d;

      dataset.push_back({
        .features = std::move(features),
        .label = label,
      });
    }
    return dataset;
  };

  Predictors predictors({
    .num_features = num_features,
    .max_tree_nodes = max_tree_nodes,
    .max_tree_height = tree_depth,
  });

  auto write_image = [&](int idx) -> bee::OrError<bee::Unit> {
    auto fast_trees = predictors.create_fast_trees();
    auto pred_one = [&](const vector<float>& features) {
      return fast_trees.eval(features);
    };
    auto start = Time::monotonic();
    bail(
      image,
      bee::FileWriter::create(bee::FilePath::of_string(
        bee::format("image-$.pnm", zero_pad(idx, 3)))));

    image->write(bee::format("P5\n$ $\n255\n", dim * 2 + 1, dim * 2 + 1));
    uint64_t num_preds = 0;
    for (int x = -dim; x <= dim; ++x) {
      string line;
      for (int y = -dim; y <= dim; ++y) {
        vector<float> features = {float(x), float(y)};
        num_preds++;
        auto p = pred_one(features);
        auto pred = std::clamp(int((p + 1.0) * 127), 0, 255);
        line += char(pred);
      }
      image->write(line);
    }
    auto end = Time::monotonic();

    auto ellapsed = end - start;
    print_line("Write image took $", ellapsed);
    print_line(
      "Time per prediction $us", ellapsed.to_float_micros() / num_preds);

    return bee::ok();
  };

  auto test_dataset = make_dataset(test_dataset_size);

  for (int i = 0; i < iters; i++) {
    print_line("-------------------------------------");
    print_line("Nodes: $", predictors.num_nodes());
    print_line("Total nodes: $", predictors.total_nodes());
    print_line("Step $", i);
    print_line("Test loss: $", predictors.loss(test_dataset));
    auto start = Time::monotonic();
    auto train_dataset = make_dataset(training_batch_size);
    predictors.train_epoch(train_dataset, num_chunks, learning_rate);
    predictors.maybe_split();
    bail_unit(write_image(i));
    auto ending = Time::monotonic();
    print_line("Took: $", ending - start);
  }

  return bee::ok();
}

command::Cmd train_kagle_command()
{
  using namespace command;
  using namespace command::flags;
  auto builder = CommandBuilder("Train");
  auto training_filename = builder.required("--training-file", string_flag);
  return builder.run([=]() { return train_kagle_main(*training_filename); });
}

command::Cmd train_image_command()
{
  using namespace command;
  using namespace command::flags;
  auto builder = CommandBuilder("Train");
  return builder.run([=]() { return train_image_main(); });
}

} // namespace
} // namespace ml

int main(int argc, char* argv[])
{
  using namespace command;
  return GroupBuilder("ML")
    .cmd("train-kaggle", ml::train_kagle_command())
    .cmd("train-image", ml::train_image_command())
    .build()
    .main(argc, argv);
}
