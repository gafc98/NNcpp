#include <iostream>
#include <string>
#include <Eigen/Dense>
#include "include/NN.cpp"
 
using Matrix = Eigen::MatrixXf;
using Vector = Eigen::VectorXf;

int main()
{
  Layer lr;
  lr.activations = Vector(12);
  lr.deltas = Vector(12);
  lr.non_linearity_type = "logistic";
  lr.W = Matrix(12, 20);

  Matrix A(3,3);
  A << 1, 2, 3, 4, 5, 6, 7, 8, 9;

  Vector v(3);
  v << 1, 2, 3;

  std::cout << A * v << "\n";
}