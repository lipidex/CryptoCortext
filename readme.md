# Dependency general
sudo apt  install cmake
sudo apt install build-essential

# Dependency HElib
sudo apt install patchelf
sudo apt install m4

# Repository dependency for CryptoCortex
git clone https://github.com/nlohmann/json.git
git clone https://github.com/llohse/libnpy.git
git clone https://github.com/homenc/HElib.git

# If you want to use helx intel acceleration for HElib
./helx_compile.sh

# Compile and install HElib
./helib_compile.sh

# Compile CryptoCortex
./cryptocortex_compile.sh