#include <encrypted-dataset.h>
#include <plain-dataset.h>
#include <config.h>

EncryptedDataset::EncryptedDataset(helib::Ctxt data) : _data(data) {}

std::vector<double> EncryptedDataset::get_data()
{
    return std::vector<double>(1, 1);
}

helib::Ctxt EncryptedDataset::get_ctxt()
{
    return _data;
}

ElementDataset& EncryptedDataset::operator*=(const ElementDataset& other)
{
    if (typeid(other) == typeid(PlainDataset))
    {
        const PlainDataset& _other = static_cast<const PlainDataset&>(other);

        helib::Ctxt el = _data;
        el *= _other._data;

        _data = el;
    }

    else if (typeid(other) == typeid(EncryptedDataset))
    {
        const EncryptedDataset& _other = static_cast<const EncryptedDataset&>(other);

        helib::Ctxt el = _data;
        el *= _other._data;

        _data = el;
    }
    
    else
        throw std::invalid_argument("The two values are not compatible for this kind of operation!");

    return *this;
}

ElementDataset& EncryptedDataset::operator+=(const ElementDataset& other)
{
    if (typeid(other) == typeid(PlainDataset))
    {
        const PlainDataset& _other = static_cast<const PlainDataset&>(other);

        helib::Ctxt el = _data;
        el += _other._data;

        _data = el;
    }

    else if (typeid(other) == typeid(EncryptedDataset))
    {
        const EncryptedDataset& _other = static_cast<const EncryptedDataset&>(other);

        helib::Ctxt el = _data;
        el += _other._data;

        _data = el;
    }
    
    else
        throw std::invalid_argument("The two values are not compatible for this kind of operation!");

    return *this;
}

ElementDataset* EncryptedDataset::clone(){
    return new EncryptedDataset(*(this));
}