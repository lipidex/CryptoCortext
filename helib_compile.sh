# git clone https://github.com/homenc/HElib.git

cd HElib
rm -rf build
mkdir build
cd build

# No HEXL
cmake -DPACKAGE_BUILD=ON ..

# HEXL
# cmake -HELIB_DEBUG=ON -DPACKAGE_BUILD=OFF -DBUILD_SHARED=OFF -DUSE_INTEL_HEXL=ON ..

make -j96

sudo make install

