#include <batch-dataset.h>

BatchDataset::BatchDataset(std::vector<std::vector<std::vector<ElementDataset*>>> data) : _data(data) {}

void BatchDataset::execute(simple_lambda operation)
{
    // std::vector<std::vector<std::vector<ElementDataset*>>> tmp = operation(_data);
    // std::vector<double> els = _data[0][0][0]->get_data();
    
    // _data = tmp;
    _data = operation(_data);
}

std::vector<std::vector<std::vector<ElementDataset*>>> BatchDataset::get_raw_data()
{
    return _data;
}

/*
BatchDataset::~BatchDataset()
{
    for (int i = 0; i < _data.size(); i++)
    {
        for (int j = 0; j < _data[i].size(); j++)
        {
            for (int k = 0; k < _data[i][j].size(); k++)
            {
                delete _data[i][j][k];
            }
            _data[i][j].clear();
        }

        _data[i].clear();
    }
}
*/