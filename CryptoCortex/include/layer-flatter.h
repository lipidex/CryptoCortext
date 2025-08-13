#ifndef LAYERFLATTER_H
#define LAYERFLATTER_H

#include <vector>
#include <cmath>

#include <helib/helib.h>

#include <layer.h>
#include <he-ops.h>

class Flatter : public Layer {
private:
    HEops* _heops;
    bool _squared = false;
    std::vector<std::vector<helib::PtxtArray>> weights;
    std::vector<helib::PtxtArray> biases;

public:
    Flatter();

    void calculate(BatchDataset& input) override;
};

#endif // LAYERFLATTER_H