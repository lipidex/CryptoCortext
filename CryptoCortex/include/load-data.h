#ifndef LOADDATA_H
#define LOADDATA_H

#include <vector>
#include <cmath>

#include <helib/helib.h>

#include <he-ops.h>
#include <batch-dataset.h>

class LoadData {
private:
    static std::vector<ElementDataset*> load_kernel(HEops heops, char* filename, size_t size);

    static std::vector<ElementDataset*> load_bias(HEops heops, char* filename);

public:
    static std::vector<std::vector<ElementDataset*>> load_dense_kernel(HEops heops, int level, size_t rows, size_t cols);
    
    static std::vector<std::vector<std::vector<std::vector<ElementDataset*>>>> load_conv_kernel(HEops heops, int level, size_t kernel_rows, size_t kernel_cols, size_t in_channels, size_t out_channels);

    static std::vector<ElementDataset*> load_dense_bias(HEops heops, int level);

    static std::vector<ElementDataset*> load_conv_bias(HEops heops, int level);

    static std::vector<ElementDataset*> load_poly_consts(HEops heops, int level);

    static std::vector<BatchDataset> load_dataset_x(HEops heops, const char* filename, size_t batch_size, size_t rows, size_t cols, size_t channels);

    static std::vector<long> load_dataset_y(const char* filename, size_t rows);
};

#endif // LOADDATA_H
