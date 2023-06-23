#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>
#include "../include/NN.cpp"
#include <algorithm>
#include <random>
 
using Matrix = Eigen::MatrixXf;
using Vector = Eigen::VectorXf;

using namespace feed_forward_net;

void simple_classifier()
{
    FF_net net;
    net.add_layer(10);
    for (size_t i = 0; i < 1; i++)
    {
        net.add_layer(30, "tanh");
    }
    net.add_layer(10, "softmax");
    net.generate_layers();
  
    std::vector<Data> training_data;

    for (size_t i = 0; i < 10; i++)
    {
        Data data;
        data.x = Vector::Zero(10);
        data.y = Vector::Zero(10);
        data.x(i) = 1.0;
        data.y(i) = 1.0;
        training_data.push_back(data);
    }
    net.set_learning_rate(0.1);

    auto rng = std::default_random_engine{};

    for (size_t e = 0; e < 10000; e++)
    {
        std::shuffle(std::begin(training_data), std::end(training_data), rng);
        float cum_loss = 0;
        for (Data & data : training_data)
        {
        net.backprop(data.x, data.y);
        cum_loss += net.get_loss();
        net.update();
        }
        std::cout << "epoch: " << e << "\tloss: " << cum_loss << "\n";
        //std::cout << "example:\nx:\n" << training_data[0].x << "\n\ny:\n" << training_data[0].y << "\n\nprediction:\n" << net.feed_forward(training_data[0].x) << "\n";
    }

    for (size_t i = 0; i < 10; i++)
        std::cout << "example no." << i << ":\nx:\n" << training_data[i].x << "\n\ny:\n" << training_data[i].y << "\n\nprediction:\n" << net.feed_forward(training_data[i].x) << "\n";
}

void simple_regression()
{
    FF_net net;
    net.add_layer(2);
    for (size_t i = 0; i < 1; i++)
    {
        net.add_layer(30, "tanh");
    }
    net.add_layer(2);
    net.generate_layers();
  
    std::vector<Data> training_data;

    for (size_t i = 0; i < 20000; i++)
    {
        Data data;
        data.x = Vector::Random(2);
        data.y = Vector(2);
        data.y << sin(data.x[0]) + cos(data.x[1]) - exp(- data.x[0] * data.x[0] - data.x[1] * data.x[1]),
                exp(data.x[1]) * tanh(data.x(0));
        training_data.push_back(data);
    }

    auto rng = std::default_random_engine{};

    for (size_t e = 0; e < 1000; e++)
    {
        std::shuffle(std::begin(training_data), std::end(training_data), rng);
        float cum_loss = 0;
        for (Data & data : training_data)
        {
        net.backprop(data.x, data.y);
        cum_loss += net.get_loss();
        net.update();
        }
        std::cout << "epoch: " << e << "\tloss: " << cum_loss << "\n";
        //std::cout << "example:\nx:\n" << training_data[0].x << "\n\ny:\n" << training_data[0].y << "\n\nprediction:\n" << net.feed_forward(training_data[0].x) << "\n";
    }

    for (size_t i = 0; i < 10; i++)
        std::cout << "example no." << i << ":\nx:\n" << training_data[i].x << "\n\ny:\n" << training_data[i].y << "\n\nprediction:\n" << net.feed_forward(training_data[i].x) << "\n";
}

void read_data(size_t n_input_samples, size_t n_output_samples, string fname, std::vector<Data> & data_vec)
{
	std::vector<string> row;
	string line, word;
 
	std::fstream file (fname, std::ios::in);
	if (file.is_open())
	{
		while (getline(file, line))
		{
			row.clear();
            Data d;
            d.x = Vector(n_input_samples);
            d.y = Vector(n_output_samples);
 
			std::stringstream str(line);

            for (size_t i = 0; i < n_input_samples; i++)
            {
                getline(str, word, ',');
                d.x(i) = std::stof(word);
                //std::cout << word << "\n";
            }
            for (size_t i = 0; i < n_output_samples; i++)
            {
                getline(str, word, ',');
                d.y(i) = std::stof(word);
                //std::cout << word << "\n";
            }
            data_vec.push_back(d);
		}
        file.close();
	}
	else
		std::cout<<"Could not open the file\n";
}

size_t get_max_idx(const Vector & v)
{
    size_t idx;
    float curr_max = - std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < v.size(); i++)
    {
        if (v(i) > curr_max)
        {
            curr_max = v(i);
            idx = i;
        }
    }
    return idx;
}

void mnist_digit_classifier()
{
    size_t n_inputs = 28 * 28;

    FF_net net;
    net.add_layer(n_inputs);
    net.add_layer(800, "leaky_ReLU");
    net.add_layer(700, "leaky_ReLU");
    net.add_layer(500, "leaky_ReLU");
    net.add_layer(400, "leaky_ReLU");
    net.add_layer(300, "leaky_ReLU");
    net.add_layer(200, "leaky_ReLU");
    net.add_layer(100, "leaky_ReLU");
    net.add_layer(10, "softmax");
    net.generate_layers();
    net.set_learning_rate(0.01);

    std::vector<Data> training_data;
    read_data(n_inputs, 1, "example/training_data.txt", training_data);
    // the target data needs to be modified to a vector that has 1 on the correct classification
    for (Data & d : training_data)
    {
        Vector new_y = Vector::Zero(10); // 10 because 10 digits
        new_y((uint8_t)d.y[0]) = 1.0;
        d.y = new_y;
    }

    auto rng = std::default_random_engine{};

    for (size_t e = 0; e < 50; e++)
    {
        std::shuffle(std::begin(training_data), std::end(training_data), rng);
        float cum_loss = 0;
        for (Data & data : training_data)
        {
        net.backprop(data.x, data.y);
        cum_loss += net.get_loss();
        net.update();
        }
        std::cout << "epoch: " << e << "\tloss: " << cum_loss << "\n";
        //std::cout << "example:\nx:\n" << training_data[0].x << "\n\ny:\n" << training_data[0].y << "\n\nprediction:\n" << net.feed_forward(training_data[0].x) << "\n";
    }

    // test examples on training and test data
    std::vector<Data> test_data;
    read_data(n_inputs, 1, "example/test_data.txt", test_data);
    for (Data & d : test_data)
    {
        Vector new_y = Vector::Zero(10); // 10 because 10 digits
        new_y((uint8_t)d.y[0]) = 1.0;
        d.y = new_y;
    }

    size_t bingos = 0;
    for (size_t i = 0; i < training_data.size(); i++)
    {
        size_t target = get_max_idx(training_data[i].y) + 1;
        size_t prediction = get_max_idx(net.feed_forward(training_data[i].x)) + 1;
        if (target == prediction)
            bingos++;
    }
    std::cout << "Success rate on training data over " << training_data.size() << " samples: " << ((float)bingos / (float)training_data.size()) * 100 << " %\n";

    bingos = 0;
    for (size_t i = 0; i < test_data.size(); i++)
    {
        size_t target = get_max_idx(test_data[i].y) + 1;
        size_t prediction = get_max_idx(net.feed_forward(test_data[i].x)) + 1;
        if (target == prediction)
            bingos++;
    }
    std::cout << "Success rate on test data over " << test_data.size() << " samples: " << ((float)bingos / (float)test_data.size()) * 100 << " %\n";

    // some examples
    for (size_t i = 0; i < 10; i++)
    {
        size_t target = get_max_idx(test_data[i].y) + 1;
        size_t prediction = get_max_idx(net.feed_forward(test_data[i].x)) + 1;
        std::cout << "example no." << i + 1 <<":\ttarget: " << target << "\tprediction: " << prediction << "\t";
        if (target == prediction)
            std::cout << "Correct!\n";
        else
            std::cout << "Wrong! :(\n";
    }
}