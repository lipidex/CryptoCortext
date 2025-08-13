#ifndef LAYERACT_H
#define LAYERACT_H

#include <vector>
#include <cmath>

#include <helib/helib.h>

#include <layer.h>
#include <he-ops.h>

class Act : public Layer {
private:
    HEops* _heops;
    std::vector<ElementDataset*> consts;

public:
    Act();

    Act(HEops* _heops, int level);

    void calculate(BatchDataset& input) override;
};

#endif // LAYERACT_H