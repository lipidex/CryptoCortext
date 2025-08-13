cd ./work
rm -rf build
mkdir build
cd build

cmake -DCMAKE_BUILD_TYPE=Debug .. ..
make -j96

# cp -r ../../model ./bin
# mv -f ./bin/HEmnist ../../