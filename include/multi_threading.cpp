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

int heavy_func()
{
	cout <<"This is a sleep_for demonstration\n";
	std::this_thread::sleep_for(std::chrono::milliseconds(5000));
	cout <<"I slept for 5000 milliseconds\n";
	return 0;
}


void test_func()
{
    std::vector<std::thread> threads;
    const uint8_t max_threads = 4;

    for (uint8_t i = 0; i < max_threads; i++)
    {
        std::thread th(heavy_func);
        threads.push_back(move(th));
    }

    cout << "waiting for threads to finish\n";

    for (auto & th  : threads)
    {
        th.join();
    }
}



#endif // MULTI_THREADING