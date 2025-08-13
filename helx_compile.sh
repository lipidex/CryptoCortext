git clone https://github.com/intel/hexl --branch v1.2.5

cd hexl

cmake -S . -B build/ 
# -DCMAKE_INSTALL_PREFIX=/usr/local/HEXL/lib/cmake/hexl-1.2.5

cmake --build build -j 96

sudo cmake --install build
