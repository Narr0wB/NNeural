
#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>

typedef float FP32;
typedef double FP64;
typedef uint32_t I32;
typedef uint64_t I64;

enum class DataType {
    FP32 = 0, FP64, I32, I64
};

template<typename T> struct TypeID;

#define ENABLE_TYPENAME(x) template<> struct TypeID<x> { static const DataType Get() { return DataType::x; } };
#define TYPE_TO_ENUM(x) TypeID<x>::Get()

ENABLE_TYPENAME(FP32);
ENABLE_TYPENAME(FP64);
ENABLE_TYPENAME(I32);
ENABLE_TYPENAME(I64);

#endif // TYPES_H