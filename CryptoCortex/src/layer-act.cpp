#include <iostream>
#include <vector>

#include <layer-act.h>
#include <config.h>
#include <load-data.h>
#include <operations.h>

Act::Act() {}

Act::Act(HEops* heops, int level)
{
    _heops = heops;

    consts = LoadData::load_poly_consts(*_heops, level);
}

void Act::calculate(BatchDataset& input)
{
    HELIB_NTIMER_START(tm_dense);
    
    Operations::apply_poly(input, consts);
    
    HELIB_NTIMER_STOP(tm_dense);
    if (tm_verbose)
        helib::printNamedTimer(std::cout, "tm_dense");
}