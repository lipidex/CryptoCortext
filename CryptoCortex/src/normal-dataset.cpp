#include <normal-dataset.h>
#include <config.h>

NormalDataset::NormalDataset(std::vector<double> data) : _data(data) {}

std::vector<double> NormalDataset::get_data()
{
    return _data;
}

ElementDataset& NormalDataset::operator*=(const ElementDataset& other)
{
    const NormalDataset& _other = static_cast<const NormalDataset&>(other);

    // if (_data.size() != _other._data.size())
    //     throw std::invalid_argument("The two values have different size!");
    
    for (size_t i=0; i<_data.size(); i++)
    {
        _data[i] *= _other._data[i];
    }

    return *this;
}

ElementDataset& NormalDataset::operator+=(const ElementDataset& other)
{
    const NormalDataset& _other = static_cast<const NormalDataset&>(other);

    // if (_data.size() != _other._data.size())
    //     throw std::invalid_argument("The two values have different size!");
    
    for (size_t i=0; i < _data.size(); i++)
    {
        _data[i] += _other._data[i];
    }
   
    return *this;
}

ElementDataset* NormalDataset::clone(){
    ElementDataset* el = new NormalDataset(*(this));
    return el;
}

NormalDataset::~NormalDataset() {
    // _data.clear();
    std::vector<double>().swap(_data);
}