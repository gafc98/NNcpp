#ifndef FEED_FORWARD_NET
#define FEED_FORWARD_NET

#include <iostream>
#include <string>
#include <Eigen/Dense>
 
using Matrix = Eigen::MatrixXf;
using Vector = Eigen::VectorXf;

struct Layer {
  Vector activations;
  std::string non_linearity_type;
  Matrix W;
  Vector deltas;
};

#endif // FEED_FORWARD_NET