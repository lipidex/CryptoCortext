#include <load-data.h>
#include <element-dataset.h>
#include <normal-dataset.h>
#include <plain-dataset.h>
#include <encrypted-dataset.h>
#include <config.h>
#include <filesystem>

#include <npy.hpp>

std::vector<ElementDataset*> LoadData::load_kernel(HEops heops, char* filename, size_t size)
{   
    std::vector<float> _weights = npy::read_npy<float>(filename).data;

    size_t _size = _weights.size();
    printf("Size: %ld\n", _size);

    if (_size != size)
    {
        throw std::invalid_argument("Invalid reshape dimensions");
    }

    std::vector<ElementDataset*> weights;
    for (int i=0; i<_weights.size(); i++)
    {
        ElementDataset* el;
        if (enable_helib)
            el = static_cast<ElementDataset*>(new PlainDataset(heops.plaintext(std::vector<double>(batch_size, static_cast<double>(_weights[i])))));
        else
            el = static_cast<ElementDataset*>(new NormalDataset(std::vector<double>(batch_size, static_cast<double>(_weights[i]))));
        
        weights.push_back(el);
    }

    return weights;
}

std::vector<std::vector<ElementDataset*>> LoadData::load_dense_kernel(HEops heops, int level, size_t rows, size_t cols)
{
    printf("Dense kernel %d\n", level+1);

    char filename[50];
    sprintf(filename, "model/dense_%d_kernel.npy", level);
    printf("Filename: %s\n", filename);
    
    std::vector<ElementDataset*> _weights = load_kernel(heops, filename, rows * cols);

    std::vector<std::vector<ElementDataset*>> weights;
    for (int i=0; i<rows; i++)
    {
        std::vector<ElementDataset*> col_weights;

        for (int j=0; j<cols; j++)
        {
            col_weights.push_back(_weights[i*cols + j]);
        }

        weights.push_back(col_weights);
    }

    printf("Shape: %ld %ld\n", weights.size(), weights[0].size());
    std::cout << std::endl;

    return weights;
}

std::vector<std::vector<std::vector<std::vector<ElementDataset*>>>> LoadData::load_conv_kernel(HEops heops, int level, size_t kernel_rows, size_t kernel_cols, size_t in_channels, size_t out_channels)
{
    printf("Conv kernel %d\n", level+1);

    char filename[50];
    sprintf(filename, "model/conv2d_%d_kernel.npy", level);
    printf("Filename: %s\n", filename);
    
    std::vector<ElementDataset*> _weights = load_kernel(heops, filename, kernel_rows * kernel_cols * in_channels * out_channels);

    std::vector<std::vector<std::vector<std::vector<ElementDataset*>>>> weights;
    for (int i=0; i<kernel_rows; i++)
    {
        std::vector<std::vector<std::vector<ElementDataset*>>> row_weights;

        for (int j=0; j<kernel_cols; j++)
        {
            std::vector<std::vector<ElementDataset*>> col_weights;

            for (int k=0; k<in_channels; k++)
            {
                std::vector<ElementDataset*> out_c_weights;

                // printf("%f %f\n", _weights[0]->get_data()[0], _weights[64]->get_data()[0]);

                for (int l=0; l<out_channels; l++)
                {
                    size_t index = i*kernel_cols*in_channels*out_channels + j*in_channels*out_channels + k*out_channels + l;
                    // printf("%ld\n", index);
                    out_c_weights.push_back(_weights[index]);
                }
                // printf("%f %f\n", out_c_weights[0]->get_data()[0], out_c_weights[1]->get_data()[0]);

                col_weights.push_back(out_c_weights);
            }

            row_weights.push_back(col_weights);
        }

        weights.push_back(row_weights);
    }

    printf("Shape: %ld %ld %ld %ld\n", weights.size(), weights[0].size(), weights[0][0].size(), weights[0][0][0].size());
    std::cout << std::endl;

    // printf("%f %f %f %f\n", weights[0][0][0][0]->get_data()[0], weights[0][0][0][1]->get_data()[0], weights[0][0][1][0]->get_data()[0], weights[0][0][1][1]->get_data()[0]);

    return weights;
}

std::vector<ElementDataset*> LoadData::load_bias(HEops heops, char* filename)
{
    std::vector<float> _npy_d = npy::read_npy<float>(filename).data;

    std::vector<ElementDataset*> _biases;
    for (int i=0; i<_npy_d.size(); i++)
    {
        ElementDataset* el;
        if (enable_helib)
            el = static_cast<ElementDataset*>(new PlainDataset(heops.plaintext(std::vector<double>(batch_size, static_cast<double>(_npy_d[i])))));
        else
            el = static_cast<ElementDataset*>(new NormalDataset(std::vector<double>(batch_size, static_cast<double>(_npy_d[i]))));
        
        _biases.push_back(el);
    }

    printf("Size: %ld\n", _biases.size());
    std::cout << std::endl;

    return _biases;
}

std::vector<ElementDataset*> LoadData::load_poly(HEops heops, char* filename)
{
    std::vector<double> _npy_d = npy::read_npy<double>(filename).data;

    std::vector<ElementDataset*> _consts;
    for (int i=0; i<_npy_d.size(); i++)
    {
        ElementDataset* el;
        if (enable_helib)
            el = static_cast<ElementDataset*>(new PlainDataset(heops.plaintext(std::vector<double>(batch_size, _npy_d[i]))));
        else
            el = static_cast<ElementDataset*>(new NormalDataset(std::vector<double>(batch_size, _npy_d[i])));
        
        _consts.push_back(el);
    }

    printf("Size: %ld\n", _consts.size());
    std::cout << std::endl;

    return _consts;
}

std::vector<ElementDataset*> LoadData::load_domain(HEops heops, char* filename)
{
    std::vector<ElementDataset*> _domain;
    // Check if file exists
    if (!std::filesystem::exists(filename))
    {
        printf("No domain available!");
        return _domain; 
    }

    std::vector<double> _npy_d = npy::read_npy<double>(filename).data;
    
    for (int i=0; i<_npy_d.size(); i++)
    {
        ElementDataset* el;
        if (enable_helib)
            el = static_cast<ElementDataset*>(new PlainDataset(heops.plaintext(std::vector<double>(batch_size, _npy_d[i]))));
        else
            el = static_cast<ElementDataset*>(new NormalDataset(std::vector<double>(batch_size, _npy_d[i])));
        
        _domain.push_back(el);
    }

    printf("Size: %ld\n", _domain.size());
    std::cout << std::endl;

    return _domain;
}

std::vector<ElementDataset*> LoadData::load_vars(HEops heops, char* filename)
{
    std::vector<float> _npy_d = npy::read_npy<float>(filename).data;

    std::vector<ElementDataset*> _vars;
    for (int i=0; i<_npy_d.size(); i++)
    {
        ElementDataset* el;
        if (enable_helib)
            el = static_cast<ElementDataset*>(new PlainDataset(heops.plaintext(std::vector<double>(batch_size, static_cast<double>(_npy_d[i])))));
        else
            el = static_cast<ElementDataset*>(new NormalDataset(std::vector<double>(batch_size, static_cast<double>(_npy_d[i]))));
        
        _vars.push_back(el);
    }

    printf("Size: %ld\n", _vars.size());
    std::cout << std::endl;

    return _vars;
}

std::vector<ElementDataset*> LoadData::load_dense_bias(HEops heops, int level)
{
    printf("Dense bias %d\n", level+1);

    char filename[50];
    sprintf(filename, "model/dense_%d_bias.npy", level);
    printf("Filename: %s\n", filename);

    return load_bias(heops, filename);
}

std::vector<ElementDataset*> LoadData::load_conv_bias(HEops heops, int level)
{
    printf("Conv bias %d\n", level+1);

    char filename[50];
    sprintf(filename, "model/conv2d_%d_bias.npy", level);
    printf("Filename: %s\n", filename);

    return load_bias(heops, filename);
}

std::vector<ElementDataset*> LoadData::load_poly_consts(HEops heops, int level)
{
    printf("Poly consts %d\n", level+1);

    char filename[50];
    sprintf(filename, "model/poly_consts_%d.npy", level);
    printf("Filename: %s\n", filename);

    return load_poly(heops, filename);
}

std::vector<ElementDataset*> LoadData::load_poly_domain(HEops heops, int level)
{
    printf("Poly domain %d\n", level+1);

    char filename[50];
    sprintf(filename, "model/poly_domain_%d.npy", level);
    printf("Filename: %s\n", filename);

    return load_domain(heops, filename);
}

std::vector<ElementDataset*> LoadData::load_batch_beta(HEops heops, int level)
{
    printf("Batch beta %d\n", level+1);

    char filename[50];
    sprintf(filename, "model/batch_normalization_%d_beta.npy", level);
    printf("Filename: %s\n", filename);

    return load_vars(heops, filename);
}

std::vector<ElementDataset*> LoadData::load_batch_gamma(HEops heops, int level)
{
    printf("Batch gamma %d\n", level+1);

    char filename[50];
    sprintf(filename, "model/batch_normalization_%d_gamma.npy", level);
    printf("Filename: %s\n", filename);

    return load_vars(heops, filename);
}

std::vector<ElementDataset*> LoadData::load_batch_mean(HEops heops, int level)
{
    printf("Batch mean %d\n", level+1);

    char filename[50];
    sprintf(filename, "model/batch_normalization_%d_moving_mean.npy", level);
    printf("Filename: %s\n", filename);

    return load_vars(heops, filename);
}

std::vector<ElementDataset*> LoadData::load_batch_variance(HEops heops, int level)
{
    printf("Batch variance %d\n", level+1);

    char filename[50];
    sprintf(filename, "model/batch_normalization_%d_moving_variance.npy", level);
    printf("Filename: %s\n", filename);

    return load_vars(heops, filename);
}


std::vector<BatchDataset> LoadData::load_dataset_x(HEops heops, const char* filename, size_t batch_size, size_t rows, size_t cols, size_t channels)
{
    std::vector<float> input = npy::read_npy<float>(filename).data;

    size_t size = input.size();
    printf("Size: %ld\n", size);

    size_t num_el = size/(rows*cols*channels);
    printf("Num elements: %ld\n", num_el);

    if (size != num_el * rows * cols * channels)
    {
        throw std::invalid_argument("Invalid reshape dimensions");
    }

    std::vector<BatchDataset> ds;

    size_t n_batch = std::ceil(static_cast<float>(num_el)/static_cast<float>(batch_size));

    for (size_t i=0; i<n_batch; i++)
    {
        std::vector<std::vector<std::vector<ElementDataset*>>> row_values;

        for (size_t j=0; j<rows; j++)
        {
            std::vector<std::vector<ElementDataset*>> col_values;

            for (size_t k=0; k<cols; k++)
            {
                std::vector<ElementDataset*> channel_values;

                for (size_t m=0; m<channels; m++)
                {
                    std::vector<double> values(batch_size, 0);

                    for (size_t l=0; l<batch_size && batch_size*i+l<num_el; l++)
                    {
                        values[l] = static_cast<double>(input[i*batch_size*(rows*cols*channels) + l*(rows*cols*channels) + j*rows*channels + k*channels + m]);
                    }

                    ElementDataset* el;
                    if (enable_helib)
                        if (enable_enc)
                            el = static_cast<ElementDataset*>(new EncryptedDataset(heops.encrypt(values)));
                        else
                            el = static_cast<ElementDataset*>(new PlainDataset(heops.plaintext(values)));
                    else
                        el = static_cast<ElementDataset*>(new NormalDataset(values));
                    
                    channel_values.push_back(el);
                }

                col_values.push_back(channel_values);
            }

            row_values.push_back(col_values);
        }

        ds.push_back(BatchDataset(row_values));
    }

    return ds;
}

std::vector<long> LoadData::load_dataset_y(const char* filename, size_t rows)
{
    std::vector<long> input = npy::read_npy<long>(filename).data;
    // printf("%s\n", typeid(input).name());

    size_t size = input.size();
    printf("Size: %ld\n", size);

    if (size != rows)
    {
        throw std::invalid_argument("Invalid reshape dimensions");
    }

    return input;
}