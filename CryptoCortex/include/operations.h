#ifndef OPERATIONS_H
#define OPERATIONS_H

#include <vector>
#include <layer.h>
#include <batch-dataset.h>

class Operations {
public:
    static void matrix_multiply(BatchDataset& input, std::vector<std::vector<ElementDataset*>> weights);
    
    static void add_vectors(BatchDataset& input, std::vector<ElementDataset*> biases);

    static void square_product(BatchDataset& input);

    static void poly_relu(BatchDataset& input);

    static void apply_poly(BatchDataset& input, std::vector<ElementDataset*> consts, std::vector<ElementDataset*> domain);

    static void batch_normalization(BatchDataset& input, std::vector<ElementDataset*>& mean, std::vector<ElementDataset*>& variance, std::vector<ElementDataset*>& gamma, std::vector<ElementDataset*>& beta, double epsilon = 1e-5);

    static void avg_pooling(BatchDataset& input, size_t filter_rows, size_t filter_cols, size_t stride_rows, size_t stride_cols, ElementDataset* divisor);

    static void matrix_conv_multiply(BatchDataset& input, std::vector<std::vector<std::vector<std::vector<ElementDataset*>>>> weights, size_t stride_rows, size_t stride_cols);

    static void add_conv_vectors(BatchDataset& input, std::vector<ElementDataset*> biases);

private:
    static std::vector<std::vector<ElementDataset*>> polys;
};

#endif // OPERATIONS_H
