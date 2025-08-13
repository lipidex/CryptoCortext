#include <iostream>
#include <vector>

#include <layer-avg.h>
#include <config.h>
#include <load-data.h>
#include <operations.h>
#include <normal-dataset.h>
#include <plain-dataset.h>

Avg::Avg(HEops* heops, size_t filter_rows, size_t filter_cols) : 
    Avg(heops, filter_rows, filter_cols, filter_rows, filter_cols)
{}

Avg::Avg(HEops* heops, size_t filter_rows, size_t filter_cols, size_t stride_rows, size_t stride_cols)
{
    _heops = heops;
    _filter_rows = filter_rows;
    _filter_cols = filter_cols;
    _stride_rows = stride_rows;
    _stride_cols = stride_cols;

    std::vector<double> data_divisor = std::vector<double>(batch_size, 1./(_filter_rows * _filter_cols));

    if (enable_helib)
        _divisor = new PlainDataset(_heops->plaintext(data_divisor));
    else
        _divisor = new NormalDataset(data_divisor);
}

void Avg::calculate(BatchDataset& input)
{
    HELIB_NTIMER_START(tm_dense);
    
    Operations::avg_pooling(input, _filter_rows, _filter_cols, _stride_rows, _stride_cols, _divisor);
    
    HELIB_NTIMER_STOP(tm_dense);
    if (tm_verbose)
        helib::printNamedTimer(std::cout, "tm_dense");
}