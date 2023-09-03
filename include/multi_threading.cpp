#ifndef MULTI_THREADING
#define MULTI_THREADING

#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <vector>
#include <map>

#include <chrono>
#include <thread>
#include "NN.cpp"
 
using Matrix = Eigen::MatrixXf;
using Vector = Eigen::VectorXf;
using std::string, std::cout;

using namespace feed_forward_net;

int heavy_func(int th_no)
{
	cout <<"This is a sleep_for demonstration\n";
    int ms_to_sleep = 5000 + th_no * 1000;
	std::this_thread::sleep_for(std::chrono::milliseconds(ms_to_sleep));
	cout <<"Thread #" << th_no << " slept for " << ms_to_sleep << " milliseconds\n";
	return 0;
}


void test_func()
{
    std::vector<std::thread> threads;
    const uint8_t max_threads = 4;

    for (uint8_t i = 0; i < max_threads; i++)
    {
        std::thread th(heavy_func, i);
        threads.push_back(move(th));
    }

    cout << "waiting for threads to finish\n";

    for (auto & th  : threads)
    {
        th.join();
    }
}

float parallel_backprop(FF_net & original_net, std::vector<Vector> x, std::vector<Vector> target)
{
    if (x.size() != target.size())
        throw std::invalid_argument("x and target do not match in size.\n");

    const unsigned int n_threads = x.size(); // create one thread for each data point
    std::vector<std::thread> threads; // threads vector
    std::vector<FF_net> nets(n_threads, original_net); // initialize nets vector with copies of original net
    
    // start parallel backprop
    for (unsigned int i; i < n_threads; i++)
    {
        std::thread th( &FF_net::backprop, &(nets[i]), x[i], target[i]);
        threads.push_back(move(th));
    }

    // wait for threads to finish
    for (auto & th : threads)
    {
        th.join();
    }

    // sum over all jacobians and get cumulative loss
    float cum_loss = 0;
    for (auto & net: nets)
    {
        cum_loss += net.get_loss();
        //cout << cum_loss << std::endl;
        original_net.sum_jacobians(net.get_layers_jac_ptr());
    }
    
    return cum_loss;
}


/*
class multi_threading_FF_net : FF_net
{
public:
    void parallel_backprop(std::vector<Vector> x, std::vector<Vector> target)
    {
        if (x.size() != target.size())
            throw std::invalid_argument("x and target do not match in size.\n");

        const uint32_t n_threads = x.size();
        std::vector<std::thread> threads(n_threads);
        std::vector<Layer> * lrs_ptr = get_layers_ptr();

        for (uint32_t i = 0; i < n_threads; i++)
        {
            std::thread th(heavy_func, i);
            threads.push_back(move(th));
        }

    };
private:
    void backprop_with_lrs_ptr(std::vector<Layer> * lrs_ptr, Vector x, Vector target)
    {
        feed_forward(x);

        std::vector<Layer> & layers = *lrs_ptr;

        size_t size_layers = layers.size();

        _layers[size_layers - 1].jac_z_b += ( layers[size_layers - 1].a - target ).cwiseProduct(deriv_func_map[_ns[size_layers - 1].non_linearity_type](_layers[size_layers - 1].a)); // derivative of loss
        _layers[size_layers - 1].jac_W += layers[size_layers - 1].jac_z_b * (layers[size_layers - 2].a.transpose());

        for (size_t i = size_layers - 1; i > 1; i--)
        {
            _layers[i - 1].jac_z_b += ( _layers[i].W.transpose() * _layers[i].jac_z_b ).cwiseProduct( deriv_func_map[_ns[i-1].non_linearity_type](_layers[i-1].a) );
            _layers[i - 1].jac_W += _layers[i - 1].jac_z_b * (_layers[i - 2].a.transpose());
        }
    };
} */

#endif // MULTI_THREADING