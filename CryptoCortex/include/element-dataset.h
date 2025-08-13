#ifndef ELEMENTDATASET_H
#define ELEMENTDATASET_H

#include <vector>

class ElementDataset {
public:
    virtual std::vector<double> get_data() = 0;
    
    virtual ElementDataset& operator*=(const ElementDataset& other) = 0;

    virtual ElementDataset& operator+=(const ElementDataset& other) = 0;

    virtual ElementDataset* clone() = 0;

    virtual ~ElementDataset() = default;
};

#endif // ELEMENTDATASET_H
