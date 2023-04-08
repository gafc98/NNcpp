
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
  net.add_layer(4, "ReLU");
  net.add_layer(4);
  net.add_layer(4, "ReLU");
  net.add_layer(4);
  net.add_layer(4, "ReLU");
  net.add_layer(4);
  net.add_layer(3);
  net.add_layer(1, "none");
  net.generate_layers();
 
  Vector x(2);
  x << 1, 2;

  Vector y(1);
  y << 7;

  for (size_t i = 0; i < 2000; i++)
  {
    net.feed_forward(x);
    std::cout <<"loss: " << net.get_loss(y) << "\n";
    //net.print_layers_activations();
    net.layer_jacobian(x, y);
    //net.print_layers_jacobian();
    net.update_net();
  }
  
}
