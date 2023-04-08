#ifndef FEED_FORWARD_NET
#define FEED_FORWARD_NET

#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <vector>
 
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

  string non_linearity_type;
};

float sigmoid(const float z)
{
  return 1.0 / (1.0 + exp(-z));
}
// Assuming z are already results of sigmoid function,
// if not it should be return Sigmoid(z) * (1.0 - Sigmoid(z));
float sigmoid_derivative(const float z)
{
  return z * (1.0 - z);
}

class FF_net
{
public:
  void add_layer(uint16_t n_neurons, string type = "logistic")
  {
    _ns.push_back(n_neurons);
  };

  void generate_layers()
  {
    Layer l;
    _layers.push_back(l); // this will be used for 1st input
    for (size_t i = 1; i < _ns.size(); i++)
    {
      l.W = _layer_init_multiplier * Matrix::Random(_ns[i], _ns[i-1]);
      l.b = _layer_init_multiplier * Vector::Random(_ns[i]);
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

  void feed_forward(Vector x)
  {
    _layers[0].a = x; // set first layer activations to input
    for (size_t i = 1; i < _layers.size(); i++)
    {
      _layers[i].z = _layers[i].W * _layers[i - 1].a + _layers[i].b;
      _layers[i].a = _func_of_vec(_layers[i].z, &sigmoid);
    } 
  };

  Vector get_loss(Vector y)
  {
    Vector dist = _layers[_layers.size() - 1].a - y;
    return dist.cwiseProduct(dist);
  };

  void layer_jacobian(Vector x, Vector y)
  {
    size_t size_layers = _layers.size();

    _layers[size_layers - 1].jac_z = _layers[size_layers - 1].a - y; // derivative of loss
    _layers[size_layers - 1].jac_b = _layers[size_layers - 1].jac_z;
    _layers[size_layers - 1].jac_W = _layers[size_layers - 1].jac_z * (_layers[size_layers - 2].a.transpose());

    for (size_t i = size_layers - 1; i > 1; i--)
    {
      _layers[i - 1].jac_z = ( _layers[i].W.transpose() * _layers[i].jac_z ).cwiseProduct(_func_of_vec(_layers[i-1].a, &sigmoid_derivative));
      _layers[i - 1].jac_b = _layers[i - 1].jac_z;
      _layers[i - 1].jac_W = _layers[i - 1].jac_z * (_layers[i - 2].a.transpose());
    }
  };

  void update_net()
  {
    for (auto & l : _layers)
    {
      l.b += - _learning_rate * l.jac_b;
      l.W += - _learning_rate * l.jac_W;
    }
  };

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

  std::vector<uint16_t> _ns;
  std::vector<Layer> _layers;
  float _layer_init_multiplier = 1.0;
  float _learning_rate = 0.001;
};

#endif // FEED_FORWARD_NET