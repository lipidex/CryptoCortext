#include <plain-dataset.h>
#include <config.h>

PlainDataset::PlainDataset(helib::PtxtArray data) : _data(data) {}

std::vector<double> PlainDataset::get_data()
{
    std::vector<double> _stored_data;
    _data.store(_stored_data);

    return _stored_data;
}

ElementDataset& PlainDataset::operator*=(const ElementDataset& other)
{
    const PlainDataset& _other = static_cast<const PlainDataset&>(other);

    helib::PtxtArray el = _data;
    el *= _other._data;

    _data = el;

    return *this;
}

ElementDataset& PlainDataset::operator+=(const ElementDataset& other)
{
    const PlainDataset& _other = static_cast<const PlainDataset&>(other);
    
    helib::PtxtArray el = _data;
    el += _other._data;

    _data = el;

    return *this;
}

ElementDataset* PlainDataset::clone(){
    return new PlainDataset(*(this));
}