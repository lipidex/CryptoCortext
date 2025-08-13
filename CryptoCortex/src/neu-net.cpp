#include <neu-net.h>
#include <config.h>

#include <nlohmann/json.hpp>
#include <npy.hpp>

#include <omp.h>


NeuralNetwork::NeuralNetwork(HEops* _heops, std::vector<Layer*> _layers)
{
    heops = _heops;

    // std::cout << "Current Working Directory: " << std::filesystem::current_path().string() << std::endl;
    printf("Loading Neural Network model...\n\n");
    
    layers = _layers;

    printf("Model loaded!\n");
}

std::vector<std::vector<float>> NeuralNetwork::reshape_output(std::vector<std::vector<double>> input)
{
    std::vector<std::vector<float>> output;

    for (size_t i=0; i<input[0].size(); i++)
    {
        std::vector<float> element;

        for (size_t j=0; j<input.size(); j++)
        {
            element.push_back(static_cast<float>(input[j][i]));
        }

        output.push_back(element);
    }

    return output;
}

std::vector<float> NeuralNetwork::softmax(const std::vector<float>& input)
{
    std::vector<float> result;
    float sum_exp = 0.0;

    // Calculate the sum of exponentials of input elements
    for (float value : input)
    {
        sum_exp += std::exp(value);
    }

    // Calculate the softmax values for each element
    for (float value : input)
    {
        float softmax_value = std::exp(value) / sum_exp;
        result.push_back(softmax_value);
    }

    return result;
}

void NeuralNetwork::predict(BatchDataset& input, int num)
{
    HELIB_NTIMER_START(tm_pred);
    
    int level = 0;
    for (auto& layer : layers)
    {
        // dump_elements(input, level, num);
        level++;

        printf("Calculate layer %d of batch %d!\n", level, (num+1));

        layer->calculate(input);

        printf("Ended layer %d of batch %d!\n", level, (num+1));

        /* if (he_verbose)
        {
            std::cout << "c.capacity=" << data[0].capacity() << " ";
            std::cout << "c.errorBound=" << data[0].errorBound() << "\n";
        }
        */
    }

    // dump_elements(input, level, num);

    HELIB_NTIMER_STOP(tm_pred);
    if (tm_verbose)
        helib::printNamedTimer(std::cout, "tm_pred");
}

std::vector<long> NeuralNetwork::get_result(BatchDataset input)
{
    std::vector<std::vector<double>> _output = heops->decrypt(input.get_raw_data()[0][0]);
    std::vector<std::vector<float>> output = reshape_output(_output);

    std::vector<long> predicted_classes;
    for (size_t i=0; i<output.size(); i++)
    {
        // Softmax activation
        // std::vector<float> result = softmax(output[i]);
        std::vector<float> result = output[i];

        long predicted_class = std::distance(result.begin(), std::max_element(result.begin(), result.end()));

        predicted_classes.push_back(predicted_class);
    }

    return predicted_classes;
}

void NeuralNetwork::dump_elements(BatchDataset ds, int level, int num)
{
    nlohmann::json jsonData;

    std::vector<std::vector<std::vector<ElementDataset*>>> els = ds.get_raw_data();

    std::vector<std::vector<std::vector<std::vector<double>>>> _extr_data;
    
    for (size_t i=0; i<els.size(); i++)
    {
        std::vector<std::vector<std::vector<double>>> _extr_data_l1;

        for (size_t j=0; j<els[i].size(); j++)
        {
            std::vector<std::vector<double>> _extr_data_l2;

            for (size_t k=0; k<els[i][j].size(); k++)
            {
                std::vector<double> _els = els[i][j][k]->get_data();

                _extr_data_l2.push_back(std::vector<double>(&_els[0], &_els[10]));
            }

            _extr_data_l1.push_back(_extr_data_l2);
        }

        _extr_data.push_back(_extr_data_l1);
    }

    std::vector<std::vector<std::vector<std::vector<double>>>> data;

    for (size_t i=0; i<_extr_data[0][0][0].size(); i++)
    {
        // if (level ==3)
        //     printf(".");
        
        std::vector<std::vector<std::vector<double>>> row;

        for (size_t j=0; j<_extr_data.size(); j++)
        {
            std::vector<std::vector<double>> col;

            for (size_t k=0; k<_extr_data[j].size(); k++)
            {
                std::vector<double> channel;

                for (size_t l=0; l<_extr_data[j][k].size(); l++)
                {
                    // double x = els[j][k][l]->get_data()[i];
                    channel.push_back(_extr_data[j][k][l][i]);
                }

                col.push_back(channel);
            }

            row.push_back(col);
        }
        
        data.push_back(row);
    }

    jsonData["data"] = data;

    char filename[50];
    sprintf(filename, "dump/level%d_%d.json", level, num);

    std::ofstream outputFile(filename);
    outputFile << jsonData.dump(4); // Pretty print with 4 spaces
    outputFile.close();

    jsonData.erase("data");

    printf("Filename: %s\n", filename);
}