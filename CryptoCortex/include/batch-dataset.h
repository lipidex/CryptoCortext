#ifndef BATCHDATASET_H
#define BATCHDATASET_H

#include <vector>
#include <functional>
#include <helib/helib.h>
#include <element-dataset.h>

typedef std::function<std::vector<std::vector<std::vector<ElementDataset*>>>(std::vector<std::vector<std::vector<ElementDataset*>>>)> simple_lambda;

class BatchDataset {
private:
    std::vector<std::vector<std::vector<ElementDataset*>>> _data;

public:
    BatchDataset(std::vector<std::vector<std::vector<ElementDataset*>>> data);

    void execute(simple_lambda operation);

    std::vector<std::vector<std::vector<ElementDataset*>>> get_raw_data();

    // ~BatchDataset();
};

#endif // BATCHDATASET_H
