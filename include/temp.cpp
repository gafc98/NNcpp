void mnist_digit_classifier_parallel()
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

    const uint8_t batch_size = 20;

    for (size_t e = 0; e < 50; e++)
    {
        std::shuffle(std::begin(training_data), std::end(training_data), rng);
        float cum_loss = 0;

        /*for (auto i = training_data.begin(); i + batch_size < training_data.end(); i = i + batch_size)
        {
            std::vector<Data> batch(i, i + batch_size);
        }*/
        
        for (size_t i = 0; i + batch_size <= training_data.size(); i = i + batch_size)
        {
            std::vector<Vector> xs, targets;
            for (size_t j = 0; j < batch_size; j++)
            {
                xs.push_back(training_data[i + j].x);
                targets.push_back(training_data[i + j].y);
            }

            //cout << xs.size() << std::endl;

            cum_loss += parallel_backprop(net, xs, targets);

            net.update();
        }


        /*for (Data & data : training_data)
        {
            net.backprop(data.x, data.y);
            cum_loss += net.get_loss();
            net.update();
        }*/
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
