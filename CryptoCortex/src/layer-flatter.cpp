#include <iostream>
#include <vector>

#include <layer-flatter.h>
#include <config.h>
#include <load-data.h>

Flatter::Flatter() {}

void Flatter::calculate(BatchDataset& input)
{
    HELIB_NTIMER_START(tm_dense);

    simple_lambda op = [](std::vector<std::vector<std::vector<ElementDataset*>>> data)
    {
        std::vector<std::vector<std::vector<ElementDataset*>>> output;
        std::vector<std::vector<ElementDataset*>> sub_output;

        std::vector<ElementDataset*> flatted;

        for (size_t i=0; i<data.size(); i++)
        {

            for (size_t j=0; j<data[i].size(); j++)
            {

                for (size_t k=0; k<data[i][j].size(); k++)
                {
                    flatted.push_back(data[i][j][k]);
                }
            }
        }

        sub_output.push_back(flatted);
        output.push_back(sub_output);

        return output;
    };

    input.execute(op);
    
    HELIB_NTIMER_STOP(tm_dense);
    if (tm_verbose)
        helib::printNamedTimer(std::cout, "tm_dense");
}