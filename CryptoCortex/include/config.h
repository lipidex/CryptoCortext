#ifndef CONFIG_H
#define CONFIG_H

#include <cstddef>

// Debug variables
inline bool tm_verbose = false;
inline bool he_verbose = false;
inline bool main_verbose = false;

// Batch size
inline size_t batch_size = 4096 * 2;

// HElib
inline bool enable_helib = true;
inline bool enable_enc = true;

#endif // CONFIG_H
