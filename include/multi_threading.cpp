#ifndef MULTI_THREADING
#define MULTI_THREADING

#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <vector>
#include <map>

#include <chrono>
#include <thread>
 
using Matrix = Eigen::MatrixXf;
using Vector = Eigen::VectorXf;
using std::string, std::cout;

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



#endif // MULTI_THREADING