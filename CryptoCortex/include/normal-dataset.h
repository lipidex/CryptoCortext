#ifndef NORMALDATASET_H
#define NORMALDATASET_H

#include <vector>
#include <element-dataset.h>

class NormalDataset : public ElementDataset {
private:
    std::vector<double> _data;

public:
    NormalDataset(std::vector<double> data);

    std::vector<double> get_data() override;

    ElementDataset& operator*=(const ElementDataset& other) override;

    ElementDataset& operator+=(const ElementDataset& other) override;

    ElementDataset* clone() override;

    ~NormalDataset() override;
};

#endif // NORMALDATASET_H