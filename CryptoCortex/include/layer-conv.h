#ifndef LAYERCONV_H
#define LAYERCONV_H

#include <vector>
#include <cmath>

#include <helib/helib.h>

#include <layer.h>
#include <he-ops.h>

class Conv : public Layer {
private:
    HEops* _heops;
    bool _relu = false;
    bool _squared = false;
    std::vector<std::vector<std::vector<std::vector<ElementDataset*>>>> weights; // (kernel_row, kernel_col, channel_in, channel_out)
    std::vector<ElementDataset*> biases;

public:
    Conv();

    Conv(HEops* heops, int level, size_t kernel_rows, size_t kernel_cols, size_t in_channels, size_t out_channels, bool relu = false, bool squared = false);

    void calculate(BatchDataset& input) override;
};

#endif // LAYERCONV_H