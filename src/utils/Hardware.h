
#ifndef HARDWARE_H
#define HARDWARE_H 

#include <type_traits>

#define MAX_THREADS 8

// Enable function declaration based on template arguments
template <bool B>
using EnableIf = typename std::enable_if<B, int>::type;

enum class Hardware {
    CPU = 0,
    GPU,
};

#endif // HARDWARE_H