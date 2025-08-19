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
    size_t _stride_rows;
    size_t _stride_cols;
    std::vector<std::vector<std::vector<std::vector<ElementDataset*>>>> weights; // (kernel_row, kernel_col, channel_in, channel_out)
    std::vector<ElementDataset*> biases;

public:
    Conv();

    Conv(HEops* heops, int level, size_t kernel_rows, size_t kernel_cols, size_t in_channels, size_t out_channels, size_t stride_rows = 1, size_t stride_cols = 1);

    void calculate(BatchDataset& input) override;
};

#endif // LAYERCONV_H