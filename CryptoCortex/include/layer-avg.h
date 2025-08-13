#ifndef LAYERAVG_H
#define LAYERAVG_H

#include <vector>
#include <cmath>

#include <helib/helib.h>

#include <layer.h>
#include <he-ops.h>

class Avg : public Layer {
private:
    HEops* _heops;
    size_t _filter_rows;
    size_t _filter_cols;
    size_t _stride_rows;
    size_t _stride_cols;
    ElementDataset* _divisor;

public:
    Avg(HEops* heops, size_t filter_rows, size_t filter_cols);

    Avg(HEops* heops, size_t filter_rows, size_t filter_cols, size_t stride_rows, size_t stride_cols);

    void calculate(BatchDataset& input) override;
};

#endif // LAYERAVG_H