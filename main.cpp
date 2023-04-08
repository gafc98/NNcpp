
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

  std::cout << "\n\n";
  Vector y(1);
  y << 0.7;
  std::cout << x;
  std::cout << "\n\n";

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

/*

#include <iostream>
#include <cmath>
#include <eigen/dense>

double Sigmoid(const double z) {
    return 1.0 / (1.0 + exp(-z));
}
// Assuming z are already results of sigmoid function,
// if not it should be return Sigmoid(z) * (1.0 - Sigmoid(z));
double SigmoidDerivative(const double z) {
    return z * (1.0 - z);
}

class NeuralNetwork {
public:
    NeuralNetwork(const Eigen::MatrixXd &x, const Eigen::VectorXd &y);
    void Feedforward();
    void BackPropagation();
    Eigen::VectorXd predicted_output;
private:
    double GetLossValue() const;
    uint32_t num_samples_;
    uint32_t num_features_;
    Eigen::MatrixXd input_;
    Eigen::VectorXd real_output_;
    // Assuming we have 2 layers
    // Neuron count per layer
    const uint32_t neuron_count_1_ = 4;
    const uint32_t neuron_count_2_ = 1;
    // Weight matrixes per layer
    // Assuming biases are 0
    Eigen::MatrixXd weights_1_;
    Eigen::MatrixXd weights_2_;
    // Temporary variable store output from first layer, changes after every feedforward
    // size will be rows - number of input samples, cols - number of neurons in 1st layer
    Eigen::MatrixXd output_1_;
};

NeuralNetwork::NeuralNetwork(const Eigen::MatrixXd & x, const Eigen::VectorXd & y) {
    input_ = x;
    real_output_ = y;
    num_samples_ = (uint32_t)input_.rows();
    num_features_ = (uint32_t)input_.cols();
    assert(num_samples_ == real_output_.rows());
    predicted_output = Eigen::VectorXd::Zero(num_samples_);
    // Random seed
    //srand((unsigned int)time(0));
    weights_1_ = Eigen::MatrixXd::Random(num_features_, neuron_count_1_);
    weights_2_ = Eigen::MatrixXd::Random(neuron_count_1_, neuron_count_2_);
    std::cout << "layer 1 weights:\n " << weights_1_ << std::endl;
    std::cout << "layer 2 weights:\n " << weights_2_ << std::endl;
}

// Move through layers to final prediction
void NeuralNetwork::Feedforward() {
    // Layer 1
    // Take all samples at once and calculate dot product for all layer 1 neurons (cols)
    // for each sample (rows)
    Eigen::MatrixXd z_1 = input_ * weights_1_;
    //std::cout << "layer 1 outuput numerical:\n " << z_1 << std::endl;
    // Squishify values in interval 0-1 using Sigmoid function
    output_1_ = z_1.unaryExpr(std::ref(Sigmoid));
    //std::cout << "layer 1 outuput:\n " << output_1_ << std::endl;
    // Layer 
    // Calculate dot product to get a predicted vector where each row is for input sample
    Eigen::VectorXd z_2 = output_1_ * weights_2_;
    //std::cout << "layer 2 outuput sum:\n " << z_2 << std::endl;
    // Squish using Sigmoid function
    predicted_output = z_2.unaryExpr(std::ref(Sigmoid));
    //std::cout << "layer 2 outuput:\n " << predicted_output << std::endl;
}

// Loss function - sum of squared errors: sum ( real-predicted )^2
double NeuralNetwork::GetLossValue() const {
    // get difference vector
    Eigen::VectorXd loss_vector = real_output_ - predicted_output;
    // Square every element
    loss_vector = loss_vector.unaryExpr([](double x) { return x * x; });
    // sum them all up
    double loss_value = loss_vector.sum();
    std::cout << "Loss value: " << loss_value << std::endl;
    return loss_value;
}

// Go back and update weights in layers from last to first
// Loss function - sum of squared errors: sum ( real-predicted )^2
// Derivative of loss function = 2 * (real-predicted) * sigmoid_der * input
void NeuralNetwork::BackPropagation() {
     GetLossValue();
    // Using derivative of loss function, find the amount to which reduce weights
    // if we move to direction of loss minimum
    // Layer 2 weight adjustment delta.
    Eigen::VectorXd delta_output = real_output_ - predicted_output;
    // Sigmoid derivative
    Eigen::VectorXd predicted_output_der = predicted_output.unaryExpr(std::ref(SigmoidDerivative));
    // Weight delt to adjust for each weight in layer 2
    Eigen::VectorXd delta_w_2 = output_1_.transpose() * (2 * delta_output.cwiseProduct(predicted_output_der));
    //std::cout << "layer 2 delta w:\n " << delta_w_2 << std::endl;
    // Layer 1 weight adjustment delta.
    // Calculate like in previous layer but uglier because there is function inside function that need to be calculated
    Eigen::MatrixXd delta_w_1 = input_.transpose() * (2 * delta_output.cwiseProduct(predicted_output_der) * weights_2_.transpose()).cwiseProduct(output_1_.unaryExpr(std::ref(SigmoidDerivative)));
    //std::cout << "layer 1 delta w:\n " << delta_w_1 << std::endl;
    // Update
    weights_1_ += delta_w_1;
    weights_2_ += delta_w_2;
}



int main() {
    const uint32_t num_samples = 4;
    const uint32_t num_features = 3;
    // Input samples
    Eigen::MatrixXd input(num_samples, num_features);
    input << 0.0, 0.0, 1.0,
             0.0, 1.0, 1.0,
             1.0, 0.0, 1.0,
             1.0, 1.0, 1.0;
    // Real outputs (column vector)
    Eigen::VectorXd output(num_samples);
    output << 0, 1, 1, 0;

    NeuralNetwork net(input, output);
    for (int i = 0; i < 15; i++) {
        net.Feedforward();
        net.BackPropagation();
    }

    std::cout << "Predicted output:\n " << net.predicted_output << std::endl;
    return 0;
}
*/