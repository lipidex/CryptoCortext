#include <iostream>
#include <vector>

#include <layer-conv.h>
#include <config.h>
#include <load-data.h>
#include <operations.h>

Conv::Conv() {}

Conv::Conv(HEops* heops, int level, size_t kernel_rows, size_t kernel_cols, size_t in_channels, size_t out_channels, size_t stride_rows, size_t stride_cols)
{
    _heops = heops;
    _stride_rows = stride_rows;
    _stride_cols = stride_cols;

    weights = LoadData::load_conv_kernel(*_heops, level, kernel_rows, kernel_cols, in_channels, out_channels);
    biases = LoadData::load_conv_bias(*_heops, level);
}

void Conv::calculate(BatchDataset& input)
{
    HELIB_NTIMER_START(tm_dense);
    
    Operations::matrix_conv_multiply(input, weights, _stride_rows, _stride_cols);

    Operations::add_conv_vectors(input, biases);
    
    HELIB_NTIMER_STOP(tm_dense);
    if (tm_verbose)
        helib::printNamedTimer(std::cout, "tm_dense");
}