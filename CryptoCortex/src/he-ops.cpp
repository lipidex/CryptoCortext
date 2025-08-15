#include <iostream>
#include <vector>

#include <he-ops.h>
#include <config.h>
#include <normal-dataset.h>
#include <plain-dataset.h>
#include <encrypted-dataset.h>

HEops::HEops()
{
    // Cyclotomic polynomial - defines phi(m)
    long m = 4 * batch_size;
    // Number of bits of the modulus chain
    long bits = 700; // prec: 170
    // Number of bits of precision when data is encoded, encrypted, or decrypted
    long prec = 40;
    // Number of columns of Key-Switching matrix (typically 2 or 3)
    long c = 3;

    std::cout << "Initialising context object..." << std::endl;
    // Context
    context = new helib::Context(helib::ContextBuilder<helib::CKKS>()
        .m(m)
        .bits(bits)
        .precision(prec)
        .c(c)
        .build());

    context->printout();
    std::cout << std::endl;

    // Security level
    std::cout << "Security: " << context->securityLevel() << std::endl;

    // Secret key management
    std::cout << "Creating secret key..." << std::endl;
    secret_key = new helib::SecKey(*context);

    // Generate the secret key
    secret_key->GenSecKey();

    // Public key management.
    public_key = secret_key;

    // EncryptedArray of the context
    ea = &context->getEA();

    // Number of slot
    long nslots = ea->size();
    std::cout << "Number of slots: " << nslots << std::endl;
    std::cout << std::endl;
}

helib::PtxtArray HEops::plaintext(std::vector<double> data)
{
    return helib::PtxtArray(*(context), data);
}

helib::Ctxt HEops::encrypt(std::vector<double> data)
{
    helib::Ctxt res = helib::Ctxt(*(public_key));

    helib::PtxtArray p0(*(context), data);

    p0.encrypt(res);

    return res;
}

std::vector<std::vector<double>> HEops::decrypt(std::vector<ElementDataset*> encrypted_data)
{
    std::vector<std::vector<double>> res;

    for (int i=0; i<encrypted_data.size(); i++)
    {
        if (typeid(*(encrypted_data[i])) == typeid(NormalDataset) || typeid(*(encrypted_data[i])) == typeid(PlainDataset))
            res.push_back(encrypted_data[i]->get_data());
        else
        {
            EncryptedDataset* _encrypted_data = static_cast<EncryptedDataset*>(encrypted_data[i]);
            helib::PtxtArray el(*(context));
            
            el.decrypt(_encrypted_data->get_ctxt(), *(secret_key));

            std::vector<double> _res;
            el.store(_res);

            res.push_back(_res);
        }
    }

    return res;
}