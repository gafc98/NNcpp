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

namespace feed_forward_net
{

struct Data
{
  Vector x;
  Vector y;
};

struct Layer
{
  Matrix W;
  Vector b;
};

struct Layer_jac
{
  Vector a; // actiovations
  Vector z; // net
  Vector jac_z_b; // denotes the jacobian of b and z as jac(z) = jac(b)
  Matrix jac_W;
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

class SomeClass {
  int& a; // This is not an uninitialized reference. It should be initialized 
          // in the constructor. Otherwise the compiler will throw an error.
public:
  SomeClass(int& b) : a(b) {} 
};

class FF_net
{
public:
  FF_net(std::vector<Net_Structure> & ns, std::vector<Layer> & layers) : _ns(ns), _layers(layers) {};

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
    Layer_jac l_jac;
    _layers_jac.push_back(l_jac);
    for (size_t i = 1; i < _ns.size(); i++)
    {
      float multiplier = sqrt(6.0 / (_ns[i].n_neurons + _ns[i-1].n_neurons)); // based on Glorot and Bengio, 2010
      l.W = multiplier * Matrix::Random(_ns[i].n_neurons, _ns[i-1].n_neurons);
      l.b = multiplier * Vector::Random(_ns[i].n_neurons);

      l_jac.jac_W = Matrix::Zero(_ns[i].n_neurons, _ns[i-1].n_neurons);
      l_jac.jac_z_b = Vector::Zero(_ns[i].n_neurons);

      _layers.push_back(l);
      _layers_jac.push_back(l_jac);
    }
    //std::cout << "layers generated\n"; 
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
    for (auto & l_jac : _layers_jac)
      std::cout << l_jac.a << "\n\n";
  };


  void print_layers_jacobian()
  {
    size_t i = 0;
    for (auto & l_jac : _layers_jac)
    {
      std::cout << "layer: " << i++ << "\n";
      std::cout << "jac_z_b:\n" << l_jac.jac_z_b << "\n\n";

      std::cout << "jac_W:\n" << l_jac.jac_W << "\n";
    }
  }

  Vector feed_forward(Vector x)
  {
    _layers_jac[0].a = x; // set first layer activations to input
    for (size_t i = 1; i < _layers.size(); i++)
    {
      _layers_jac[i].z = _layers[i].W * _layers_jac[i - 1].a + _layers[i].b;
      _layers_jac[i].a = func_map[_ns[i].non_linearity_type](_layers_jac[i].z);
    }
    return _layers_jac[_layers.size() - 1].a;
  };

  float get_loss()
  {
    Vector dist = _layers_jac[_layers.size() - 1].a - _target;
    return dist.squaredNorm();
  };

  void backprop(Vector x, Vector target)
  {
    feed_forward(x);

    _target = target;

    size_t size_layers = _layers.size();

    _layers_jac[size_layers - 1].jac_z_b = ( _layers_jac[size_layers - 1].a - _target ).cwiseProduct(deriv_func_map[_ns[size_layers - 1].non_linearity_type](_layers_jac[size_layers - 1].a)); // derivative of loss
    _layers_jac[size_layers - 1].jac_W = _layers_jac[size_layers - 1].jac_z_b * (_layers_jac[size_layers - 2].a.transpose());

    for (size_t i = size_layers - 1; i > 1; i--)
    {
      _layers_jac[i - 1].jac_z_b = ( _layers[i].W.transpose() * _layers_jac[i].jac_z_b ).cwiseProduct( deriv_func_map[_ns[i-1].non_linearity_type](_layers_jac[i-1].a) );
      _layers_jac[i - 1].jac_W = _layers_jac[i - 1].jac_z_b * (_layers_jac[i - 2].a.transpose());
    }
  };

  void update()
  {
    for (size_t i = 0; i < _layers.size(); i++)
    {
      _layers[i].b += - _learning_rate * _layers_jac[i].jac_z_b;
      _layers[i].W += - _learning_rate * _layers_jac[i].jac_W;

      _layers_jac[i].jac_z_b.setZero();
      _layers_jac[i].jac_W.setZero();
    }
  }

  void sum_jacobians(std::vector<Layer_jac> * layers_jac)
  {
    size_t i = 0;
    for (auto & l_jac : _layers_jac)
    {
      l_jac.jac_z_b += (*(layers_jac))[i].jac_z_b;
      l_jac.jac_W += (*(layers_jac))[i].jac_W;
      i++;
    }
  }

  void set_learning_rate(float lr)
  {
    _learning_rate = lr;
  }

  std::vector<Layer_jac> * get_layers_jac_ptr()
  {
    return &_layers_jac;
  }

private:
  std::vector<Net_Structure> &_ns;
  std::vector<Layer> &_layers;
  std::vector<Layer_jac> _layers_jac;
  Vector _target;
  float _learning_rate = 0.001;
};

};

#endif // FEED_FORWARD_NET