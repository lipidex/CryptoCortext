#include <iostream>
#include <vector>

#include <layer-dense.h>
#include <config.h>
#include <load-data.h>
#include <operations.h>

Dense::Dense() {}

Dense::Dense(HEops* heops, int level, size_t rows, size_t cols, bool relu, bool squared)
{
    _heops = heops;
    _relu = relu;
    _squared = squared;

    weights = LoadData::load_dense_kernel(*_heops, level, rows, cols);
    biases = LoadData::load_dense_bias(*_heops, level);
}

void Dense::calculate(BatchDataset& input)
{
    HELIB_NTIMER_START(tm_dense);
    
    Operations::matrix_multiply(input, weights);

    Operations::add_vectors(input, biases);

    if (_relu)
        Operations::poly_relu(input);

    if (_squared)
        Operations::square_product(input);
    
    HELIB_NTIMER_STOP(tm_dense);
    if (tm_verbose)
        helib::printNamedTimer(std::cout, "tm_dense");
}