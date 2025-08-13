#ifndef CONFIG_H
#define CONFIG_H

#include <cstddef>

// Debug variables
inline bool tm_verbose = false;
inline bool he_verbose = false;
inline bool main_verbose = false;

// Batch size
inline size_t batch_size = 4096 / 4;

// HElib
inline bool enable_helib = false;
inline bool enable_enc = false;

#endif // CONFIG_H
