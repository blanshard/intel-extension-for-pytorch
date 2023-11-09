#pragma once
#include <utils/DPCPP.h>

#include <c10/util/Exception.h>
#include <utils/Macros.h>
#include <utils/oneMKLUtils.h>

namespace xpu {
namespace dpcpp {

#define AT_DPCPP_TRY try {
#ifdef USE_ONEMKL
#define AT_DPCPP_CATCH_RETHROW(filename, lineno)                               \
  }                                                                            \
  catch (oneapi::mkl::lapack::exception & e) {                                 \
    xpu::oneMKL::oneMKLExpInfo::Instance().setLastInfo(e.info());              \
    TORCH_WARN(                                                                \
        "ONEMKL Exception:",                                                   \
        e.info(),                                                              \
        "file = ",                                                             \
        filename,                                                              \
        "line = ",                                                             \
        lineno);                                                               \
    throw;                                                                     \
  }                                                                            \
  catch (sycl::exception & e) {                                                \
    TORCH_WARN(                                                                \
        "SYCL Exception: ", e.what(), "file = ", filename, "line = ", lineno); \
    throw;                                                                     \
  }

#define AT_DPCPP_CATCH_NOTHROW(filename, lineno)                               \
  }                                                                            \
  catch (oneapi::mkl::lapack::exception & e) {                                 \
    xpu::oneMKL::oneMKLExpInfo::Instance().setLastInfo(e.info());              \
    TORCH_WARN(                                                                \
        "ONEMKL Exception:",                                                   \
        e.info(),                                                              \
        "file = ",                                                             \
        filename,                                                              \
        "line = ",                                                             \
        lineno);                                                               \
  }                                                                            \
  catch (sycl::exception & e) {                                                \
    TORCH_WARN(                                                                \
        "SYCL Exception: ", e.what(), "file = ", filename, "line = ", lineno); \
  }
#else
#define AT_DPCPP_CATCH_RETHROW(filename, lineno)                               \
  }                                                                            \
  catch (sycl::exception & e) {                                                \
    TORCH_WARN(                                                                \
        "SYCL Exception: ", e.what(), "file = ", filename, "line = ", lineno); \
    throw;                                                                     \
  }

#define AT_DPCPP_CATCH_NOTHROW(filename, lineno)                               \
  }                                                                            \
  catch (sycl::exception & e) {                                                \
    TORCH_WARN(                                                                \
        "SYCL Exception: ", e.what(), "file = ", filename, "line = ", lineno); \
  }
#endif

#define __AT_DPCPP_CHECK(EXPR, filename, lineno) \
  do {                                           \
    int __err;                                   \
    AT_DPCPP_TRY                                 \
    __err = EXPR;                                \
    AT_DPCPP_CATCH_RETHROW(filename, lineno)     \
    if (__err != DPCPP_SUCCESS) {                \
      TORCH_CHECK(0, "XPU error: %d", __err);    \
    }                                            \
  } while (0)

#define __AT_DPCPP_CHECK_NORET(EXPR, filename, lineno) \
  do {                                                 \
    AT_DPCPP_TRY(EXPR);                                \
    AT_DPCPP_CATCH_RETHROW(filename, lineno)           \
  } while (0)

#define __AT_DPCPP_CHECK_WARN(EXPR, filename, lineno) \
  do {                                                \
    AT_DPCPP_TRY(EXPR);                               \
    AT_DPCPP_CATCH_NOTHROW(filename, lineno)          \
  } while (0)

#define AT_DPCPP_CHECK(EXPR) __AT_DPCPP_CHECK(EXPR, __FILE__, __LINE__)
#define AT_DPCPP_CHECK_NORET(EXPR) \
  __AT_DPCPP_CHECK_NORET(EXPR, __FILE__, __LINE__)
#define AT_DPCPP_CHECK_WARN(EXPR) \
  __AT_DPCPP_CHECK_WARN(EXPR, __FILE__, __LINE__)

#define AT_DPCPP_NOTHROW(...) \
  try {                       \
    __VA_ARGS__;              \
  } catch (...) {             \
  }

#define AT_DPCPP_ASSERT(cond)                     \
  do {                                            \
    if (!(cond)) {                                \
      TORCH_CHECK(0, "assert(%s) failed", #cond); \
    }                                             \
  } while (0)

static sycl::async_handler dpcppAsyncHandler = [](sycl::exception_list eL) {
  for (auto& e : eL) {
    AT_DPCPP_TRY
    std::rethrow_exception(e);
    AT_DPCPP_CATCH_RETHROW(__FILE__, __LINE__)
  }
};

} // namespace dpcpp
} // namespace xpu
