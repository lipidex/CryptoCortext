#ifndef ENCRYPTEDDATASET_H
#define ENCRYPTEDDATASET_H

#include <vector>
#include <element-dataset.h>

#include <helib/helib.h>

class EncryptedDataset : public ElementDataset {
private:
    helib::Ctxt _data;

public:
    EncryptedDataset(helib::Ctxt data);

    std::vector<double> get_data() override;

    helib::Ctxt get_ctxt();

    ElementDataset& operator*=(const ElementDataset& other) override;

    ElementDataset& operator+=(const ElementDataset& other) override;

    ElementDataset* clone() override;

    ~EncryptedDataset() override = default;
};

#endif // ENCRYPTEDDATASET_H