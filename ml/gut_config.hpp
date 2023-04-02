#pragma once

namespace ml {

enum class LossFunction {
  L2,
  L1,
};

struct GutConfig {
  const int num_features;

  const int max_tree_nodes = 256;
  const int max_tree_height = 16;
  const int max_num_trees = 8;

  const int min_samples_to_split = 1000;
  const double learning_rate = 1.0;
  const double lr_decay = 0.7;
  const double ew_lambda = 0.999;

  const bool update_threshold = false;

  const LossFunction loss_function = LossFunction::L2;
};

} // namespace ml
