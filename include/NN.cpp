#ifndef FEED_FORWARD_NET
#define FEED_FORWARD_NET

#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <vector>
#include <map>
 
using Matrix = Eigen::MatrixXf;
using Vector = Eigen::VectorXf;
using std::string;

struct Data
{
  Vector x;
  Vector y;
};

struct Layer
{
  Vector a; // actiovations
  Vector z; // net
  Matrix W;
  Vector b;

  Vector jac_z;
  Matrix jac_W;
  Vector jac_b;
};

struct Net_Structure
{
  string non_linearity_type;
  uint16_t n_neurons;
};

Vector tanh(Vector z)
{
  for (auto & i : z)
    i = 2.0 / (1.0 + exp(-i)) - 1.0;
  return z;
}
Vector tanh_derivative(Vector tanh)
{
  for (auto & i : tanh)
    i = 1.0 - i * i;
  return tanh;
}

Vector linear(Vector z)
{
  return z;
}
Vector linear_derivative(Vector lin)
{
  return Vector::Ones(lin.size());
}

Vector ReLU(Vector z)
{
  for (auto & i : z)
  {
    if (i <= 0.0)
      i = 0.0;
  }
  return z;
}
Vector ReLU_derivative(Vector relu)
{
  for (auto & i : relu)
  {
    if (i <= 0.0)
      i = 0.0;
    else
      i = 1.0;
  }
  return relu;
}

Vector leaky_ReLU(Vector z)
{
  for (auto & i : z)
  {
    if (i <= 0.0)
      i = 0.001 * i;
  }
  return z;
}
Vector leaky_ReLU_derivative(Vector relu)
{
  for (auto & i : relu)
  {
    if (i <= 0.0)
      i = 0.001;
    else
      i = 1.0;
  }
  return relu;
}

Vector softmax(Vector z)
{
  for (auto & i : z)
    i = exp(i);
  return z / z.sum();
}
Vector softmax_derivative(Vector smax)
{
  for (auto & i : smax)
    i = i * (1.0 - i);
  return smax;
}

std::map<std::string, Vector (*)(Vector)> func_map
{
  {"none", &linear},
  {"tanh", &tanh},
  {"ReLU", &ReLU},
  {"leaky_ReLU", &leaky_ReLU},
  {"softmax", &softmax}
};
std::map<std::string, Vector (*)(Vector)> deriv_func_map
{
  {"none", &linear_derivative},
  {"tanh", &tanh_derivative},
  {"ReLU", &ReLU_derivative},
  {"leaky_ReLU", &leaky_ReLU_derivative},
  {"softmax", &softmax_derivative}
};

class FF_net
{
public:
  void add_layer(uint16_t n_neurons, string type = "none")
  {
    Net_Structure s;
    s.n_neurons = n_neurons;
    s.non_linearity_type = type;
    _ns.push_back(s);
  };

  void generate_layers()
  {
    Layer l;
    _layers.push_back(l); // this will be used for 1st input
    for (size_t i = 1; i < _ns.size(); i++)
    {
      l.W = _layer_init_multiplier * Matrix::Random(_ns[i].n_neurons, _ns[i-1].n_neurons);
      l.b = _layer_init_multiplier * Vector::Random(_ns[i].n_neurons);
      _layers.push_back(l);
    }
  };

  void print_layers_W()
  {
    std::cout << "layers:\n";
    for (auto & l : _layers)
      std::cout << l.W << "\n\n";
  };

  void print_layers_activations()
  {
    std::cout << "activations:\n";
    for (auto & l : _layers)
      std::cout << l.a << "\n\n";
  };


  void print_layers_jacobian()
  {
    size_t i = 0;
    for (auto & l : _layers)
    {
      std::cout << "layer: " << i++ << "\n";
      std::cout << "jac_z:\n" << l.jac_z << "\n\n";
      std::cout << "jac_b:\n" << l.jac_b << "\n";
      std::cout << "jac_W:\n" << l.jac_W << "\n";
    }
  }

  Vector feed_forward(Vector x)
  {
    _layers[0].a = x; // set first layer activations to input
    for (size_t i = 1; i < _layers.size(); i++)
    {
      _layers[i].z = _layers[i].W * _layers[i - 1].a + _layers[i].b;
      _layers[i].a = func_map[_ns[i].non_linearity_type](_layers[i].z);
    }
    return _layers[_layers.size() - 1].a;
  };

  float get_loss(Vector target)
  {
    Vector dist = _layers[_layers.size() - 1].a - target;
    return dist.squaredNorm();
  };

  void update_net(Vector target)
  {
    size_t size_layers = _layers.size();

    _layers[size_layers - 1].jac_z = ( _layers[size_layers - 1].a - target ).cwiseProduct(deriv_func_map[_ns[size_layers - 1].non_linearity_type](_layers[size_layers - 1].a)); // derivative of loss
    _layers[size_layers - 1].jac_b = _layers[size_layers - 1].jac_z;
    _layers[size_layers - 1].jac_W = _layers[size_layers - 1].jac_z * (_layers[size_layers - 2].a.transpose());

    for (size_t i = size_layers - 1; i > 1; i--)
    {
      _layers[i - 1].jac_z = ( _layers[i].W.transpose() * _layers[i].jac_z ).cwiseProduct( deriv_func_map[_ns[i-1].non_linearity_type](_layers[i-1].a) );
      _layers[i - 1].jac_b = _layers[i - 1].jac_z;
      _layers[i - 1].jac_W = _layers[i - 1].jac_z * (_layers[i - 2].a.transpose());
    }

    // update weights
    for (auto & l : _layers)
    {
      l.b += - _learning_rate * l.jac_b;
      l.W += - _learning_rate * l.jac_W;
    }
  };

  void set_learning_rate(float lr)
  {
    _learning_rate = lr;
  }

private:
  std::vector<Net_Structure> _ns;
  std::vector<Layer> _layers;
  float _layer_init_multiplier = 0.01;
  float _learning_rate = 0.001;
};

#endif // FEED_FORWARD_NET