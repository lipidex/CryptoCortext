#ifndef NEUNET_H
#define NEUNET_H

#include <vector>
#include <cmath>

#include <he-ops.h>
#include <layer.h>
#include <batch-dataset.h>

class NeuralNetwork {
private:
    std::vector<Layer*> layers;
    HEops* heops;

public:
    NeuralNetwork(HEops* _heops, std::vector<Layer*> _layers);

    std::vector<std::vector<float>> reshape_output(std::vector<std::vector<double>> input);

    std::vector<float> softmax(const std::vector<float>& input);

    void predict(BatchDataset& input, int num);

    std::vector<long> get_result(BatchDataset input);

    void dump_elements(BatchDataset ds, int level, int num);

};

#endif // NEUNET_H
