
#include <iostream>
#include <string>
#include <Eigen/Dense>
#include "include/NN.cpp"
#include <algorithm>
#include <random>
 
using Matrix = Eigen::MatrixXf;
using Vector = Eigen::VectorXf;

struct Data
{
  Vector x;
  Vector y;
};

int main()
{
  FF_net net;
  net.add_layer(2);
  for (size_t i = 0; i < 1; i++)
  {
    //net.add_layer(50, "leaky_ReLU");
    net.add_layer(30, "tanh");
    net.add_layer(30, "tanh");
  }
  net.add_layer(2);
  net.generate_layers();
 
  
  std::vector<Data> training_data;
  
  for (size_t i = 0; i < 20000; i++)
  {
    Data data;
    data.x = Vector::Random(2);
    data.y = Vector(2);
    data.y << sin(data.x[0]) + cos(data.x[1]) - exp(- data.x[0] * data.x[0] - data.x[1] * data.x[1]),
              exp(data.x[1]) * tanh(data.x(0));
    training_data.push_back(data);
  }

  auto rng = std::default_random_engine{};

  for (size_t e = 0; e < 10000; e++)
  {
    std::shuffle(std::begin(training_data), std::end(training_data), rng);
    float cum_loss = 0;
    for (Data & data : training_data)
    {
      net.feed_forward(data.x);
      cum_loss += net.get_loss(data.y);
      net.update_net(data.y);
    }
    std::cout <<"loss: " << cum_loss << "\n";
    //std::cout << "example:\nx:\n" << training_data[0].x << "\n\ny:\n" << training_data[0].y << "\n\nprediction:\n" << net.feed_forward(training_data[0].x) << "\n";
  }

  
}
