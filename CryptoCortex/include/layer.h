#ifndef LAYER_H
#define LAYER_H

#include <vector>

#include <batch-dataset.h>

class Layer {
public:
    virtual void calculate(BatchDataset& input) = 0;
    virtual ~Layer() = default;
};

#endif // LAYER_H
