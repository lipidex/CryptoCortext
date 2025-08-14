#include <operations.h>
#include <config.h>
#include <normal-dataset.h>

void Operations::matrix_multiply(BatchDataset& input, std::vector<std::vector<ElementDataset*>> weights)
{
    HELIB_NTIMER_START(tm_weights);

    simple_lambda op = [weights](std::vector<std::vector<std::vector<ElementDataset*>>> data)
    {
        if (data[0][0].size() != weights.size())
        {
            throw std::invalid_argument("Input size is different from weights size!");
        }

        std::vector<std::vector<std::vector<ElementDataset*>>> output;
        std::vector<std::vector<ElementDataset*>> sub_output;

        std::vector<ElementDataset*> result;

        bool to_create = true;
        for (int i=0; i<weights.size(); i++)
        {
            for (int j=0; j<weights[i].size(); j++)
            {   
                ElementDataset* el = data[0][0][i]->clone(); // Once flattered all the elements are in position 0
                *(el) *= *(weights[i][j]);
                
                if (to_create)
                    result.push_back(el);
                else
                {
                    *(result[j]) += *(el);
                    delete el;
                }
            }

            to_create = false;

            delete data[0][0][i];
        }

        sub_output.push_back(result);
        output.push_back(sub_output);

        return output;
    };

    input.execute(op);

    HELIB_NTIMER_STOP(tm_weights);
    if (tm_verbose)
        helib::printNamedTimer(std::cout, "tm_weights");
}

void Operations::add_vectors(BatchDataset& input, std::vector<ElementDataset*> biases)
{
    HELIB_NTIMER_START(tm_biases);

    simple_lambda op = [biases](std::vector<std::vector<std::vector<ElementDataset*>>> data)
    {
        if (data[0][0].size() != biases.size())
        {
            throw std::invalid_argument("Input size is different from bias size!");
        }

        std::vector<std::vector<std::vector<ElementDataset*>>> output;
        std::vector<std::vector<ElementDataset*>> sub_output;

        std::vector<ElementDataset*> result;

        for (int i=0; i<data[0][0].size(); i++)
        {
            ElementDataset* el = data[0][0][i];

            *(el) += *(biases[i]);
            result.push_back(el);
        }

        sub_output.push_back(result);
        output.push_back(sub_output);

        return output;
    };

    input.execute(op);

    HELIB_NTIMER_STOP(tm_biases);

    if (tm_verbose)
        helib::printNamedTimer(std::cout, "tm_biases");
}

void Operations::square_product(BatchDataset& input)
{
    simple_lambda op = [](std::vector<std::vector<std::vector<ElementDataset*>>> data)
    {
        std::vector<std::vector<std::vector<ElementDataset*>>> output;

        for (size_t i=0; i<data.size(); i++)
        {
            std::vector<std::vector<ElementDataset*>> sub_output;

            for (size_t j=0; j<data[i].size(); j++)
            {
                std::vector<ElementDataset*> result;

                for (int k=0; k<data[i][j].size(); k++)
                {
                    ElementDataset* el = data[i][j][k]->clone();

                    *(el) *= *(data[i][j][k]);
                    result.push_back(el);

                    delete data[i][j][k];
                }

                sub_output.push_back(result);
            }

            output.push_back(sub_output);
        }

        return output;
    };

    input.execute(op);
}

void Operations::poly_relu(BatchDataset& input)
{
    simple_lambda op = [](std::vector<std::vector<std::vector<ElementDataset*>>> data)
    {
        std::vector<std::vector<std::vector<ElementDataset*>>> output;

        for (size_t i=0; i<data.size(); i++)
        {
            std::vector<std::vector<ElementDataset*>> sub_output;

            for (size_t j=0; j<data[i].size(); j++)
            {
                std::vector<ElementDataset*> result;

                for (int k=0; k<data[i][j].size(); k++)
                {

                    // +0.05555555588275139*x**0
                    // +0.44444444460816346*x**1
                    // +0.4444444441172486*x**2
                    ElementDataset* el = data[i][j][k]->clone();

                    double domain = 29.52;

                    std::vector<double> _dn = std::vector<double>(batch_size, 1/domain);
                    ElementDataset* dn = new NormalDataset(_dn);
                    *(el) *= *(dn);

                    ElementDataset* _base_x = el->clone();
                    ElementDataset* _x = el->clone();

                    // if x^1 exist
                    std::vector<double> _c1 = std::vector<double>(batch_size, 0.44444444460816346);
                    ElementDataset* c1 = new NormalDataset(_c1);
                    *(el) *= *(c1);

                    // if x^0 exist
                    std::vector<double> _c0 = std::vector<double>(batch_size, 0.05555555588275139);
                    ElementDataset* c0 = new NormalDataset(_c0);
                    *(el) += *(c0);

                    // for 2 to max degree
                    *(_x) *= *(_base_x);

                    ElementDataset* mon = _x->clone();

                    std::vector<double> _c2 = std::vector<double>(batch_size, 0.4444444441172486);
                    ElementDataset* c2 = new NormalDataset(_c2);

                    *(mon) *= *(c2);

                    *(el) += *(mon);
                    // end for

                    std::vector<double> _dp = std::vector<double>(batch_size, domain);
                    ElementDataset* dp = new NormalDataset(_dp);
                    *(el) *= *(dp);
                    
                    result.push_back(el);

                    delete data[i][j][k];
                }

                sub_output.push_back(result);
            }

            output.push_back(sub_output);
        }

        return output;
    };

    input.execute(op);
}

void Operations::apply_poly(BatchDataset& input, std::vector<ElementDataset*> consts)
{
    simple_lambda op = [consts](std::vector<std::vector<std::vector<ElementDataset*>>> data)
    {
        std::vector<std::vector<std::vector<ElementDataset*>>> output;

        for (size_t i=0; i<data.size(); i++)
        {
            std::vector<std::vector<ElementDataset*>> sub_output;

            for (size_t j=0; j<data[i].size(); j++)
            {
                std::vector<ElementDataset*> result;

                for (int k=0; k<data[i][j].size(); k++)
                {
                    ElementDataset* _pow_x = data[i][j][k]->clone();

                    double domain = 14.29;

                    std::vector<double> _dd = std::vector<double>(batch_size, 1/domain);
                    ElementDataset* dd = new NormalDataset(_dd);
                    *(_pow_x) *= *(dd); // Contain final result

                    ElementDataset* _base_x = _pow_x->clone(); // Contain initial value of x
                    // ElementDataset* _pow_x = el->clone(); // Contain multiple of x

                    ElementDataset* el = consts[0]->clone();
                    for (int v=1; v<consts.size(); v++)
                    {
                        if (v >= 2)
                        {
                            *(_pow_x) *= *(_base_x); // Calculate new power of x
                        }

                        ElementDataset* mon = _pow_x->clone(); // Use clone

                        *(mon) *= *(consts[v]); // Multiply with constant

                        *(el) += *(mon); // Add to result
                        delete mon; // Delete clone
                    }

                    std::vector<double> _dm = std::vector<double>(batch_size, domain);
                    ElementDataset* dm = new NormalDataset(_dm);

                    *(el) *= *(dm); // Multiply with domain

                    result.push_back(el);

                    delete data[i][j][k];
                    delete _base_x;
                    delete _pow_x;
                    delete dd;
                    delete dm;
                }

                sub_output.push_back(result);
            }

            output.push_back(sub_output);
        }

        return output;
    };

    input.execute(op);
}

void Operations::avg_pooling(BatchDataset& input, size_t filter_rows, size_t filter_cols, size_t stride_rows, size_t stride_cols, ElementDataset* divisor)
{
    simple_lambda op = [filter_rows, filter_cols, stride_rows, stride_cols, divisor](std::vector<std::vector<std::vector<ElementDataset*>>> data)
    {
        std::vector<std::vector<std::vector<ElementDataset*>>> output;

        for (size_t i=0; i+filter_rows<=data.size(); i+=stride_rows)
        {
            std::vector<std::vector<ElementDataset*>> cols;

            for (size_t j=0; j+filter_cols<=data[i].size(); j+=stride_cols)
            {
                std::vector<ElementDataset*> channels;

                for (size_t k=0; k<filter_rows; k++)
                {
                    for (size_t l=0; l<filter_cols; l++)
                    {
                        if (k == 0 && l == 0)
                            channels = data[i][j];
                        else
                            for (size_t m=0; m<data[i][j].size(); m++)
                            {
                                *(channels[m]) += *(data[i+k][j+l][m]);
                            }
                    }
                }

                for (size_t k=0; k<channels.size(); k++)
                {
                    *(channels[k]) *= *(divisor);
                }

                cols.push_back(channels);
            }

            output.push_back(cols);
        }

        return output;
    };

    input.execute(op);
}

void Operations::matrix_conv_multiply(BatchDataset& input, std::vector<std::vector<std::vector<std::vector<ElementDataset*>>>> weights, size_t stride_rows, size_t stride_cols)
{
    simple_lambda op = [weights, stride_rows, stride_cols](std::vector<std::vector<std::vector<ElementDataset*>>> data)
    {
        std::vector<std::vector<std::vector<ElementDataset*>>> output;

        size_t num_hop_row = 1 + ((data.size() - weights.size())/stride_rows);
        size_t num_hop_col = 1 + ((data[0].size() - weights[0].size())/stride_cols);
        for (size_t i=0; i<num_hop_row; i++) // number of row for each channel
        {
            std::vector<std::vector<ElementDataset*>> cols;

            for (size_t j=0; j<num_hop_col; j++) // number of col for each channel
            {
                std::vector<ElementDataset*> channel_out;

                bool already_exist = false;

                for (size_t k=0; k<weights.size(); k++) // fetch kernel rows
                {
                    for (size_t l=0; l<weights[k].size(); l++) // fetch kernel cols
                    {
                        for (size_t m=0; m<weights[k][l].size(); m++) // fetch channel in
                        {
                            for (size_t n=0; n<weights[k][l][m].size(); n++) // fetch channel out
                            {
                                ElementDataset* el = data[i+(k*stride_rows)][j+(l*stride_cols)][m]->clone();
                                // printf("%f\n", el->get_data()[0]);
                                // printf("%f\n", weights[k][l][m][n]->get_data()[0]);

                                *(el) *= *(weights[k][l][m][n]);
                                // printf("%f\n", el->get_data()[0]);
                                
                                if (already_exist)
                                {
                                    *(channel_out[n]) += *(el);
                                    // printf("%f\n", channel_out[n]->get_data()[0]);

                                    delete el;
                                }
                                else
                                    channel_out.push_back(el);
                                
                                // printf("\n");
                            }

                            already_exist = true;
                        }
                    }
                }
                // printf("%f\n", channel_out[0]->get_data()[0]);

                cols.push_back(channel_out);
            }

            output.push_back(cols);
        }
        
        for (size_t i=0; i<data.size(); i++)
        {
            for (size_t j=0; j<data[i].size(); j++)
            {
                for (size_t k=0; k<data[i][j].size(); k++)
                {
                    delete data[i][j][k];
                }
            }
        }

        // delete[][][] data;

        return output;
    };

    input.execute(op);
}

void Operations::add_conv_vectors(BatchDataset& input, std::vector<ElementDataset*> biases)
{
    simple_lambda op = [biases](std::vector<std::vector<std::vector<ElementDataset*>>> data)
    {
        std::vector<std::vector<std::vector<ElementDataset*>>> output;

        for (size_t i=0; i<data.size(); i++)
        {
            std::vector<std::vector<ElementDataset*>> cols;

            for (size_t j=0; j<data[i].size(); j++)
            {
                std::vector<ElementDataset*> channels;

                for (size_t k=0; k<data[i][j].size(); k++)
                {
                    ElementDataset* el = data[i][j][k];
                    *(el) += *(biases[k]);

                    channels.push_back(el);
                }

                cols.push_back(channels);
            }

            output.push_back(cols);
        }
        return output;
    };

    input.execute(op);
}