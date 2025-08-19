#include <iostream>
#include <vector>

#include <neu-net.h>
#include <layer-conv.h>
#include <layer-batch.h>
#include <layer-avg.h>
#include <layer-dense.h>
#include <layer-act.h>
#include <layer-flatter.h>
#include <load-data.h>
#include <config.h>

int main()
{
    HEops heops;

    std::vector<Layer*> layers;
    layers.push_back(new Conv(&heops, 0, 5, 5, 1, 5, 2, 2));
    layers.push_back(new Batch(&heops, 0));
    layers.push_back(new Act(&heops, 0));
    layers.push_back(new Avg(&heops, 2, 2));
    layers.push_back(new Flatter());
    layers.push_back(new Dense(&heops, 0, 180, 10));

    NeuralNetwork neural_network(&heops, layers);
    size_t num_el = 10000;

    // Load test dataset
    printf("Loading test_x dataset...\n");
    std::vector<BatchDataset> test_x = LoadData::load_dataset_x(heops, "model/test_x.npy", batch_size, 28, 28, 1);
    printf("Loaded test_x dataset!\n");

    printf("Loading test_y dataset...\n");
    std::vector<long> test_y = LoadData::load_dataset_y("model/test_y.npy", num_el);
    printf("Loaded test_y dataset!\n");

    size_t num_batch = test_x.size();

    std::cout << std::endl;

    int perc = 0;
    int accuracy = 0;

    // num_threads(numCores)
    // reduction(+:accuracy)
    HELIB_NTIMER_START(tm_tot_pred);
    #pragma omp parallel for schedule(dynamic) // num_threads(1)
    for (size_t i = 0; i < num_batch; i++)
    {
        printf("Prediction of batch %ld of %ld...\n", i+1, num_batch);

        neural_network.predict(test_x[i], i);
        
        std::vector<long> output = neural_network.get_result(test_x[i]);

        for (size_t j=0; j < output.size(); j++)
        {
            if (i*batch_size + j >= num_el)
                break;
            
            long predicted_class = output[j];
            if (predicted_class == test_y[i*batch_size + j])
                accuracy++;

            /*
            if (perc < 100/sample_size*(i+1))
            {
                perc = 100/sample_size*(i+1);
                std::cout << "Progress: " << perc << "%" << std::endl;
            }
            */

            if (main_verbose)
            {
                std::cout<< "Last correct class: " << test_y[i*batch_size + j] << "\n";
                std::cout<< "Last predicted class: " << predicted_class << "\n";
                printf("Predictions iterated: %ld\n", ((i+1)*batch_size + j));
                printf("Accuracy: %f%%\n", 100./((i+1)*batch_size + j)*accuracy);
                printf("Correct predictions: %d\n", accuracy);
                std::cout << std::endl;
            }
        }
    }
    HELIB_NTIMER_STOP(tm_tot_pred);
    helib::printNamedTimer(std::cout, "tm_tot_pred");

    // Result
    printf("Stats: \n");
    printf("Accuracy: %f%%\n", 100./num_el*accuracy);
    printf("Correct predictions: %d\n", accuracy);

    return 0;
}