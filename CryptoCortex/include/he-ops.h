#ifndef HEOPS_H
#define HEOPS_H

#include <helib/helib.h>

#include <batch-dataset.h>

class HEops {
private:
    helib::Context* context;
    helib::SecKey* secret_key;
    helib::PubKey* public_key;
    const helib::EncryptedArray* ea;

public:
    const std::string _model_base_path;

    HEops(long bits, long prec, long c, std::string model_base_path);

    helib::PtxtArray plaintext(std::vector<double> data);

    helib::Ctxt encrypt(std::vector<double> data);

    std::vector<std::vector<double>> decrypt(std::vector<ElementDataset*> encrypted_data);
};

#endif // HEOPS_H
