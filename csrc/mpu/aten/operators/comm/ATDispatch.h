#pragma once

#include <ATen/Dispatch.h>

#ifdef BUILD_INTERNAL_DEBUG
#define IPEX_PRIVATE_CASE_TYPE(enum_type, type, ...) \
  case enum_type: {                                  \
    using scalar_t = type;                           \
    return __VA_ARGS__();                            \
  }

#define IPEX_QINT_PRIVATE_CASE_TYPE(                          \
    enum_type, type, underlying_enum, underlying_type, ...)   \
  case enum_type: {                                           \
    const auto& UNDERLYING_TYPE C10_UNUSED = underlying_enum; \
    using scalar_t C10_UNUSED = type;                         \
    using underlying_t C10_UNUSED = underlying_type;          \
    return __VA_ARGS__();                                     \
  }

#else // NOT BUILD_INTERNAL_DEBUG

#define IPEX_PRIVATE_CASE_TYPE(enum_type, type, ...) \
  AT_DISPATCH_CASE(enum_type, __VA_ARGS__)

#endif // BUILD_INTERNAL_DEBUG

#define IPEX_DISPATCH_ATOMIC_ALL_TYPES(TYPE, NAME, ...)                      \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op  */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
    constexpr const char* at_dispatch_name = NAME;                           \
    switch (_st) {                                                           \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)      \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)      \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)      \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Half, at::Half, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Bool, bool, __VA_ARGS__)        \
      IPEX_PRIVATE_CASE_TYPE(                                                \
          at::ScalarType::BFloat16, at::BFloat16, __VA_ARGS__)               \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)     \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

#define IPEX_DISPATCH_ATOMIC_ALL_TYPES_AND_COMPLEX(TYPE, NAME, ...)          \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op  */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
    constexpr const char* at_dispatch_name = NAME;                           \
    switch (_st) {                                                           \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)      \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)      \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)      \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Half, at::Half, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Bool, bool, __VA_ARGS__)        \
      IPEX_PRIVATE_CASE_TYPE(                                                \
          at::ScalarType::BFloat16, at::BFloat16, __VA_ARGS__)               \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(                                                \
          at::ScalarType::ComplexFloat, c10::complex<float>, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(                                                \
          at::ScalarType::ComplexDouble, c10::complex<double>, __VA_ARGS__)  \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

#define IPEX_DISPATCH_ATOMIC_FLOATING_TYPES(TYPE, NAME, ...)                 \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op  */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
    constexpr const char* at_dispatch_name = NAME;                           \
    switch (_st) {                                                           \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)      \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Half, at::Half, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(                                                \
          at::ScalarType::BFloat16, at::BFloat16, __VA_ARGS__)               \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)    \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

#define IPEX_DISPATCH_ATOMIC_FLOATING_AND_COMPLEX_TYPES(TYPE, NAME, ...)     \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op  */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
    constexpr const char* at_dispatch_name = NAME;                           \
    switch (_st) {                                                           \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)      \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Half, at::Half, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(                                                \
          at::ScalarType::BFloat16, at::BFloat16, __VA_ARGS__)               \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(                                                \
          at::ScalarType::ComplexFloat, c10::complex<float>, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(                                                \
          at::ScalarType::ComplexDouble, c10::complex<double>, __VA_ARGS__)  \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

#define IPEX_DISPATCH_BYTE_ALL_TYPES_AND(SCALARTYPE, TYPE, NAME, ...)        \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op  */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
    constexpr const char* at_dispatch_name = NAME;                           \
    switch (_st) {                                                           \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)      \
      IPEX_PRIVATE_CASE_TYPE(at::kQInt8, int8_t, __VA_ARGS__)                \
      IPEX_PRIVATE_CASE_TYPE(at::kQUInt8, uint8_t, __VA_ARGS__)              \
      IPEX_PRIVATE_CASE_TYPE(                                                \
          SCALARTYPE,                                                        \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE>::t),           \
          __VA_ARGS__)                                                       \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

// Some kernels must support full data types, for example, fill and copy.
// This dispatch macro helps to keep all data types support in any build config.
#define IPEX_DISPATCH_ALL_TYPES_ALWAYS_AND(SCALARTYPE, TYPE, NAME, ...) \
  AT_DISPATCH_ALL_TYPES_AND(SCALARTYPE, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_ALL_TYPES_ALWAYS_AND2(   \
    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...) \
  AT_DISPATCH_ALL_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_ALL_TYPES_ALWAYS_AND3(                \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...) \
  AT_DISPATCH_ALL_TYPES_AND3(                               \
      SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, __VA_ARGS__)

// Redefine DISPATCH macro for debug usage only
// For example, remove useless data type support for multiple kernels to reduce
// JIT time

#ifdef BUILD_INTERNAL_DEBUG

#define IPEX_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                       \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    switch (_st) {                                                          \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)     \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      \
    }                                                                       \
  }()

#define IPEX_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, ...)              \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    switch (_st) {                                                          \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Half, at::Half, __VA_ARGS__)   \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      \
    }                                                                       \
  }()

#define IPEX_DISPATCH_FLOATING_TYPES_AND(SCALARTYPE, TYPE, NAME, ...)       \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    switch (_st) {                                                          \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE,                                                       \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE>::t),          \
          __VA_ARGS__)                                                      \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");     \
    }                                                                       \
  }()

#define IPEX_DISPATCH_FLOATING_TYPES_AND2(                                  \
    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)                              \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    switch (_st) {                                                          \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE1,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE1>::t),         \
          __VA_ARGS__)                                                      \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE2,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE2>::t),         \
          __VA_ARGS__)                                                      \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");     \
    }                                                                       \
  }()

#define IPEX_DISPATCH_FLOATING_TYPES_AND3(                                  \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...)                 \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    switch (_st) {                                                          \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE1,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE1>::t),         \
          __VA_ARGS__)                                                      \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE2,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE2>::t),         \
          __VA_ARGS__)                                                      \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE3,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE3>::t),         \
          __VA_ARGS__)                                                      \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");     \
    }                                                                       \
  }()

#define IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES(TYPE, NAME, ...)           \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    switch (_st) {                                                          \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__) \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)   \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      \
    }                                                                       \
  }()

#define IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(                      \
    SCALARTYPE, TYPE, NAME, ...)                                            \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    switch (_st) {                                                          \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__) \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE,                                                       \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE>::t),          \
          __VA_ARGS__)                                                      \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      \
    }                                                                       \
  }()

#define IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                      \
    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)                              \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    switch (_st) {                                                          \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__) \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE1,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE1>::t),         \
          __VA_ARGS__)                                                      \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE2,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE2>::t),         \
          __VA_ARGS__)                                                      \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      \
    }                                                                       \
  }()

#define IPEX_DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...)                       \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    switch (_st) {                                                          \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)   \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      \
    }                                                                       \
  }()

#define IPEX_DISPATCH_INTEGRAL_TYPES_AND(SCALARTYPE, TYPE, NAME, ...)     \
  [&] {                                                                   \
    switch (TYPE) {                                                       \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)  \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)  \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__) \
      IPEX_PRIVATE_CASE_TYPE(                                             \
          SCALARTYPE,                                                     \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE>::t),        \
          __VA_ARGS__)                                                    \
      default:                                                            \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");   \
    }                                                                     \
  }()

#define IPEX_DISPATCH_ALL_TYPES(TYPE, NAME, ...)                             \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op  */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
    switch (_st) {                                                           \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)      \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)      \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)      \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)    \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

#define IPEX_DISPATCH_COMPLEX_TYPES(TYPE, NAME, ...)                        \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    switch (_st) {                                                          \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__) \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      \
    }                                                                       \
  }()

#define IPEX_DISPATCH_QINT_TYPES(TYPE, NAME, ...)                       \
  [&] {                                                                 \
    const auto& SCALAR_TYPE C10_UNUSED = TYPE;                          \
    switch (TYPE) {                                                     \
      IPEX_QINT_PRIVATE_CASE_TYPE(                                      \
          at::kQInt8, at::qint8, at::kChar, int8_t, __VA_ARGS__)        \
      IPEX_QINT_PRIVATE_CASE_TYPE(                                      \
          at::kQUInt8, at::quint8, at::kByte, uint8_t, __VA_ARGS__)     \
      IPEX_QINT_PRIVATE_CASE_TYPE(                                      \
          at::kQInt32, at::qint32, at::kInt, int, __VA_ARGS__)          \
      default:                                                          \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'"); \
    }                                                                   \
  }()

#define IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX(TYPE, NAME, ...)                \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op*/  \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    switch (_st) {                                                          \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__) \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      \
    }                                                                       \
  }()

#define IPEX_DISPATCH_ALL_TYPES_AND(SCALARTYPE, TYPE, NAME, ...)          \
  [&] {                                                                   \
    switch (TYPE) {                                                       \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)  \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__) \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)  \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__) \
      IPEX_PRIVATE_CASE_TYPE(                                             \
          SCALARTYPE,                                                     \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE>::t),        \
          __VA_ARGS__)                                                    \
      default:                                                            \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");   \
    }                                                                     \
  }()

#define IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(SCALARTYPE, TYPE, NAME, ...) \
  [&] {                                                                      \
    switch (TYPE) {                                                          \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)      \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)      \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)      \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(                                                \
          at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(                                                \
          at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__)  \
      IPEX_PRIVATE_CASE_TYPE(                                                \
          SCALARTYPE,                                                        \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE>::t),           \
          __VA_ARGS__)                                                       \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");      \
    }                                                                        \
  }()

#define IPEX_DISPATCH_COMPLEX_TYPES_AND(SCALARTYPE, TYPE, NAME, ...)        \
  [&] {                                                                     \
    switch (TYPE) {                                                         \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__) \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE,                                                       \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE>::t),          \
          __VA_ARGS__)                                                      \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");     \
    }                                                                       \
  }()

#define IPEX_DISPATCH_ALL_TYPES_AND2(                                     \
    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)                            \
  [&] {                                                                   \
    switch (TYPE) {                                                       \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)  \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__) \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)  \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__) \
      IPEX_PRIVATE_CASE_TYPE(                                             \
          SCALARTYPE1,                                                    \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE1>::t),       \
          __VA_ARGS__)                                                    \
      IPEX_PRIVATE_CASE_TYPE(                                             \
          SCALARTYPE2,                                                    \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE2>::t),       \
          __VA_ARGS__)                                                    \
      default:                                                            \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");   \
    }                                                                     \
  }()

#define IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(                           \
    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)                              \
  [&] {                                                                     \
    switch (TYPE) {                                                         \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__) \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE1,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE1>::t),         \
          __VA_ARGS__)                                                      \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE2,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE2>::t),         \
          __VA_ARGS__)                                                      \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", TYPE, "'");               \
    }                                                                       \
  }()

#define IPEX_DISPATCH_ALL_TYPES_AND3(                                     \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...)               \
  [&] {                                                                   \
    switch (TYPE) {                                                       \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)  \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__) \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)  \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__) \
      IPEX_PRIVATE_CASE_TYPE(                                             \
          SCALARTYPE1,                                                    \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE1>::t),       \
          __VA_ARGS__)                                                    \
      IPEX_PRIVATE_CASE_TYPE(                                             \
          SCALARTYPE2,                                                    \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE2>::t),       \
          __VA_ARGS__)                                                    \
      IPEX_PRIVATE_CASE_TYPE(                                             \
          SCALARTYPE3,                                                    \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE3>::t),       \
          __VA_ARGS__)                                                    \
      default:                                                            \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");   \
    }                                                                     \
  }()

#define IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(                           \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...)                 \
  [&] {                                                                     \
    switch (TYPE) {                                                         \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          at::ScalarType::ComplexFloat, c10::complex<float>, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          at::ScalarType::ComplexDouble, c10::complex<double>, __VA_ARGS__) \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE1,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE1>::t),         \
          __VA_ARGS__)                                                      \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE2,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE2>::t),         \
          __VA_ARGS__)                                                      \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE3,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE3>::t),         \
          __VA_ARGS__)                                                      \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", TYPE, "'");               \
    }                                                                       \
  }()

#define IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(                           \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, TYPE, NAME, ...)    \
  [&] {                                                                     \
    switch (TYPE) {                                                         \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          at::ScalarType::ComplexFloat, c10::complex<float>, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          at::ScalarType::ComplexDouble, c10::complex<double>, __VA_ARGS__) \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE1,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE1>::t),         \
          __VA_ARGS__)                                                      \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE2,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE2>::t),         \
          __VA_ARGS__)                                                      \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE3,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE3>::t),         \
          __VA_ARGS__)                                                      \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE4,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE4>::t),         \
          __VA_ARGS__)                                                      \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", TYPE, "'");               \
    }                                                                       \
  }()

#else // Reuse DISPATCH macro in ATen

#define IPEX_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, ...) \
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_FLOATING_TYPES_AND(SCALARTYPE, TYPE, NAME, ...) \
  AT_DISPATCH_FLOATING_TYPES_AND(SCALARTYPE, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_FLOATING_TYPES_AND2(     \
    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...) \
  AT_DISPATCH_FLOATING_TYPES_AND2(             \
      SCALARTYPE1, SCALARTYPE2, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_CASE_FLOATING_TYPES_AND3( \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, ...) \
  AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__)  \
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)    \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)    \
  AT_DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)

#define IPEX_DISPATCH_FLOATING_TYPES_AND3(                  \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                       \
      TYPE,                                                 \
      NAME,                                                 \
      IPEX_DISPATCH_CASE_FLOATING_TYPES_AND3(               \
          SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, __VA_ARGS__))

#define IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1( \
    SCALARTYPE, TYPE, NAME, ...)                       \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(         \
      SCALARTYPE, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2( \
    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)         \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(         \
      SCALARTYPE1, SCALARTYPE2, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_INTEGRAL_TYPES(TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_INTEGRAL_TYPES_AND(SCALARTYPE, TYPE, NAME, ...) \
  AT_DISPATCH_INTEGRAL_TYPES_AND(SCALARTYPE, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_ALL_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_ALL_TYPES(TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_COMPLEX_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_COMPLEX_TYPES(TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_QINT_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_QINT_TYPES(TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX(TYPE, NAME, ...) \
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_ALL_TYPES_AND(SCALARTYPE, TYPE, NAME, ...) \
  AT_DISPATCH_ALL_TYPES_AND(SCALARTYPE, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(SCALARTYPE, TYPE, NAME, ...) \
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(SCALARTYPE, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_COMPLEX_TYPES_AND(SCALARTYPE, TYPE, NAME, ...) \
  AT_DISPATCH_COMPLEX_TYPES_AND(SCALARTYPE, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_ALL_TYPES_AND2(          \
    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...) \
  AT_DISPATCH_ALL_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2( \
    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)    \
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(         \
      SCALARTYPE1, SCALARTYPE2, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_ALL_TYPES_AND3(                       \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...) \
  AT_DISPATCH_ALL_TYPES_AND3(                               \
      SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(           \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...) \
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(                   \
      SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(                        \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, TYPE, NAME, ...) \
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(                                \
      SCALARTYPE1,                                                       \
      SCALARTYPE2,                                                       \
      SCALARTYPE3,                                                       \
      SCALARTYPE4,                                                       \
      TYPE,                                                              \
      NAME,                                                              \
      __VA_ARGS__)

#endif

#define IPEX_DISPATCH_INDEX_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_INDEX_TYPES(TYPE, NAME, __VA_ARGS__)

#define IPEX_PRIVATE_CASE_TYPE_WITH_UNDERLYING(enum_type, type, SUFFIX, ...) \
  case enum_type: {                                                          \
    using scalar_t_##SUFFIX C10_UNUSED = type;                               \
    using underlying_t_##SUFFIX C10_UNUSED = type;                           \
    using bool_t_##SUFFIX C10_UNUSED = std::false_type;                      \
    return __VA_ARGS__();                                                    \
  }

#define IPEX_QINT_PRIVATE_CASE_TYPE_WITH_UNDERLYING(                \
    enum_type, type, underlying_enum, underlying_type, SUFFIX, ...) \
  case enum_type: {                                                 \
    const auto& UNDERLYING_TYPE C10_UNUSED = underlying_enum;       \
    using scalar_t_##SUFFIX C10_UNUSED = underlying_type;           \
    using underlying_t_##SUFFIX C10_UNUSED = underlying_type;       \
    using bool_t_##SUFFIX C10_UNUSED = std::true_type;              \
    return __VA_ARGS__();                                           \
  }

#define IPEX_DISPATCH_QTYPE_WITH_UNDERLYING(TYPE, NAME, SUFFIX, ...)        \
  [&] {                                                                     \
    switch (TYPE) {                                                         \
      IPEX_PRIVATE_CASE_TYPE_WITH_UNDERLYING(                               \
          at::ScalarType::Float, float, SUFFIX, __VA_ARGS__)                \
      IPEX_QINT_PRIVATE_CASE_TYPE_WITH_UNDERLYING(                          \
          at::kQInt8, at::qint8, at::kChar, int8_t, SUFFIX, __VA_ARGS__)    \
      IPEX_QINT_PRIVATE_CASE_TYPE_WITH_UNDERLYING(                          \
          at::kQUInt8, at::quint8, at::kByte, uint8_t, SUFFIX, __VA_ARGS__) \
      IPEX_QINT_PRIVATE_CASE_TYPE_WITH_UNDERLYING(                          \
          at::kQInt32, at::qint32, at::kInt, int, SUFFIX, __VA_ARGS__)      \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", TYPE, "'");               \
    }                                                                       \
  }()

#define IPEX_DISPATCH_QTYPE_ONLY_WITH_UNDERLYING(TYPE, NAME, SUFFIX, ...)   \
  [&] {                                                                     \
    switch (TYPE) {                                                         \
      IPEX_QINT_PRIVATE_CASE_TYPE_WITH_UNDERLYING(                          \
          at::kQInt8, at::qint8, at::kChar, int8_t, SUFFIX, __VA_ARGS__)    \
      IPEX_QINT_PRIVATE_CASE_TYPE_WITH_UNDERLYING(                          \
          at::kQUInt8, at::quint8, at::kByte, uint8_t, SUFFIX, __VA_ARGS__) \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", TYPE, "'");               \
    }                                                                       \
  }()
