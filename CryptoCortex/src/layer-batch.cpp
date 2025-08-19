#include <iostream>
#include <vector>

#include <layer-batch.h>
#include <config.h>
#include <load-data.h>
#include <operations.h>

Batch::Batch() {}

Batch::Batch(HEops* heops, int level)
{
    _heops = heops;

    mean = LoadData::load_batch_mean(*_heops, level);
    variance = LoadData::load_batch_variance(*_heops, level);
    gamma = LoadData::load_batch_gamma(*_heops, level);
    beta = LoadData::load_batch_beta(*_heops, level);
}

void Batch::calculate(BatchDataset& input)
{
    HELIB_NTIMER_START(tm_dense);
    
    Operations::batch_normalization(input, mean, variance, gamma, beta);
    
    HELIB_NTIMER_STOP(tm_dense);
    if (tm_verbose)
        helib::printNamedTimer(std::cout, "tm_dense");
}