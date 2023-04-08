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
  Vector activations;
  Vector nets;
  string non_linearity_type;
  Matrix W;
  Vector deltas;
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
    for (size_t i = 1; i < _ns.size(); i++)
    {
      Layer l; 
      l.W = _layer_init_multiplier * Matrix::Random(_ns[i], _ns[i-1] + 1); // + 1 due to bias
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
      std::cout << l.activations << "\n\n";
  };

  void print_layers_deltas()
  {
    std::cout << "deltas:\n";
    for (auto & l : _layers)
      std::cout << l.deltas << "\n\n";
  };

  void feed_forward(Vector x)
  {
    _append_one_to_vec(x);

    // first layer activations
    _layers[0].nets = _layers[0].W * x;
    _layers[0].activations = _func_of_vec(_layers[0].nets, &sigmoid);
    _append_one_to_vec(_layers[0].activations);

    for (size_t i = 1; i < _layers.size(); i++)
    {
      _layers[i].nets = _layers[i].W * _layers[i-1].activations;
      _layers[i].activations = _func_of_vec(_layers[i].nets, &sigmoid);
      _append_one_to_vec(_layers[i].activations);
    }
  };

  void layer_jacobian()
  {
    // last layer
    _layers[_layers.size() - 1].deltas = _func_of_vec(_layers[_layers.size() - 1].activations, &sigmoid_derivative); //.segment(0, (_layers[_layers.size() - 1].activations).size() - 1)
    //std::cout << "deltas:\n" << _layers[_layers.size() - 1].deltas << "\n\n";
    for (size_t i = _layers.size() - 1; i > 0; i--)
    {
      size_t size_deltas = _layers[i].deltas.size();
      _layers[i - 1].deltas = (_layers[i].W).transpose() * _layers[i].deltas.segment(0, size_deltas - 1);
      //std::cout << _func_of_vec(_layers[i].activations, &sigmoid_derivative);
      _layers[i - 1].deltas = _layers[i - 1].deltas.cwiseProduct(_func_of_vec(_layers[i-1].activations, &sigmoid_derivative));
      //std::cout << _layers[i - 1].deltas << "\n\n";
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
};

#endif // FEED_FORWARD_NET