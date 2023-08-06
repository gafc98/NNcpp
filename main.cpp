
#include <iostream>
#include <string>
#include <Eigen/Dense>
#include "include/NN.cpp"
#include "include/multi_threading.cpp"
#include "example/examples.cpp"
#include <algorithm>
#include <random>
 
using Matrix = Eigen::MatrixXf;
using Vector = Eigen::VectorXf;

int main()
{
  //simple_classifier();
  //simple_regression();
  //mnist_digit_classifier();
  mnist_digit_classifier_parallel();
  test_func();
}