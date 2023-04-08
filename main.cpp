#include <iostream>
#include <string>
#include <Eigen/Dense>
#include "include/NN.cpp"
 
using Matrix = Eigen::MatrixXf;
using Vector = Eigen::VectorXf;

int main()
{
  FF_net net;
  net.add_layer(2);
  net.add_layer(4);
  net.add_layer(3);
  net.add_layer(1);
  net.generate_layers();
  net.print_layers_W();

  std::cout << "\n\n";
  Vector x(2);
  x << 1, 2;
  std::cout << x;
  std::cout << "\n\n";

  net.feed_forward(x);
  net.print_layers_activations();
  net.layer_jacobian();
  net.print_layers_deltas();
}