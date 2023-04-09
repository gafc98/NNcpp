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

float tanh(const float z)
{
  return 2.0 / (1.0 + exp(-z)) - 1.0;
}
float tanh_derivative(const float tanh)
{
  //float sig = tanh(z);
  return 1.0 - tanh * tanh;
}

float linear(const float z)
{
  return z;
}
float linear_derivative(const float lin)
{
  return 1.0;
}

float ReLU(const float z)
{
  if (z > 0)
    return z;
  return 0;
}
float ReLU_derivative(const float relu)
{
  if (relu > 0)
    return 1;
  return 0;
}

float leaky_ReLU(const float z)
{
  if (z > 0)
    return z;
  return 0.001 * z;
}
float leaky_ReLU_derivative(const float relu)
{
  if (relu > 0)
    return 1;
  return 0.001;
}

std::map<std::string, float (*)(float)> func_map
{
  {"none", &linear},
  {"tanh", &tanh},
  {"ReLU", &ReLU},
  {"leaky_ReLU", &leaky_ReLU}
};
std::map<std::string, float (*)(float)> deriv_func_map
{
  {"none", &linear_derivative},
  {"tanh", &tanh_derivative},
  {"ReLU", &ReLU_derivative},
  {"leaky_ReLU", &leaky_ReLU_derivative}
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
      _layers[i].a = _func_of_vec(_layers[i].z, func_map[_ns[i].non_linearity_type]);
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

    _layers[size_layers - 1].jac_z = _layers[size_layers - 1].a - target; // derivative of loss
    _layers[size_layers - 1].jac_b = _layers[size_layers - 1].jac_z;
    _layers[size_layers - 1].jac_W = _layers[size_layers - 1].jac_z * (_layers[size_layers - 2].a.transpose());

    for (size_t i = size_layers - 1; i > 1; i--)
    {
      _layers[i - 1].jac_z = ( _layers[i].W.transpose() * _layers[i].jac_z ).cwiseProduct(_func_of_vec(_layers[i-1].a, deriv_func_map[_ns[i-1].non_linearity_type]));
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
  inline void _append_one_to_vec(Vector & v)
  {
    v.conservativeResize(v.size() + 1);
    v[v.size() - 1] = 1.0;
  };

  inline Vector _func_of_vec(Vector v, float (func) (float))
  {
    for (float & element : v)
      element = func(element);
    return v;
  };

  std::vector<Net_Structure> _ns;
  std::vector<Layer> _layers;
  float _layer_init_multiplier = 0.01;
  float _learning_rate = 0.001;
};

#endif // FEED_FORWARD_NET