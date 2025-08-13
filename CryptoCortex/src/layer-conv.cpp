#include <iostream>
#include <vector>

#include <layer-conv.h>
#include <config.h>
#include <load-data.h>
#include <operations.h>

Conv::Conv() {}

Conv::Conv(HEops* heops, int level, size_t kernel_rows, size_t kernel_cols, size_t in_channels, size_t out_channels, bool relu, bool squared)
{
    _heops = heops;
    _relu = relu;
    _squared = squared;

    weights = LoadData::load_conv_kernel(*_heops, level, kernel_rows, kernel_cols, in_channels, out_channels);
    biases = LoadData::load_conv_bias(*_heops, level);
}

void Conv::calculate(BatchDataset& input)
{
    HELIB_NTIMER_START(tm_dense);
    
    Operations::matrix_conv_multiply(input, weights, 1, 1);

    Operations::add_conv_vectors(input, biases);

    if (_relu)
        Operations::poly_relu(input);

    if (_squared)
        Operations::square_product(input);
    
    HELIB_NTIMER_STOP(tm_dense);
    if (tm_verbose)
        helib::printNamedTimer(std::cout, "tm_dense");
}