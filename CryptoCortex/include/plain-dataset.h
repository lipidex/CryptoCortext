#ifndef PLAINDATASET_H
#define PLAINDATASET_H

#include <vector>
#include <element-dataset.h>

#include <helib/helib.h>

class PlainDataset : public ElementDataset {
//private:

public:
    helib::PtxtArray _data;
    
    PlainDataset(helib::PtxtArray data);

    std::vector<double> get_data() override;

    ElementDataset& operator*=(const ElementDataset& other) override;

    ElementDataset& operator+=(const ElementDataset& other) override;

    ElementDataset* clone() override;

    ~PlainDataset() override = default;
};

#endif // PLAINDATASET_H