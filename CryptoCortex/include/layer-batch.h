#ifndef LAYERBATCH_H
#define LAYERBATCH_H

#include <vector>
#include <cmath>

#include <helib/helib.h>

#include <layer.h>
#include <he-ops.h>

class Batch : public Layer {
private:
    HEops* _heops;
    std::vector<ElementDataset*> mean;
    std::vector<ElementDataset*> gamma_variance;
    std::vector<ElementDataset*> beta;

public:
    Batch();

    Batch(HEops* _heops, int level);

    void calculate(BatchDataset& input) override;
};

#endif // LAYERBATCH_H