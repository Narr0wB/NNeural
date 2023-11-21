
#ifndef HARDWARE_H
#define HARDWARE_H 

#include <type_traits>

// Enable function declaration based on template arguments
template <bool B>
using EnableIf = typename std::enable_if<B, int>::type;

enum class Hardware {
    CPU = 0,
    GPU,
};

#endif // HARDWARE_H