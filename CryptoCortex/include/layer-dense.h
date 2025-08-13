#ifndef LAYERDENSE_H
#define LAYERDENSE_H

#include <vector>
#include <cmath>

#include <helib/helib.h>

#include <layer.h>
#include <he-ops.h>

class Dense : public Layer {
private:
    HEops* _heops;
    bool _relu = false;
    bool _squared = false;
    std::vector<std::vector<ElementDataset*>> weights;
    std::vector<ElementDataset*> biases;

public:
    Dense();

    Dense(HEops* _heops, int level, size_t rows, size_t cols, bool relu = false, bool squared = false);

    void calculate(BatchDataset& input) override;
};

#endif // LAYERDENSE_H