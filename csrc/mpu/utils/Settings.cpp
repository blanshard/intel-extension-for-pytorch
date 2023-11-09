#include <oneDNN/Runtime.h>
#include <runtime/Device.h>
#include <utils/Settings.h>
#include <utils/oneMKLUtils.h>

#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>

namespace xpu {
namespace dpcpp {

/*
 * [Keep the format for automatic doc generation.]
 * All available launch options for IPEX
 * ==========ALL==========
 *   IPEX_FP32_MATH_MODE:
 *      Default = 0 | Set values for FP32 math mode (0: FP32, 1: TF32, 2: BF32)
 * ==========ALL==========
 *
 * XPU ONLY optionos:
 * ==========GPU==========
 *   IPEX_VERBOSE:
 *      Default = 0 | Set verbose level with synchronization execution mode
 *   IPEX_XPU_SYNC_MODE:
 *      Default = 0 | Set 1 to enforce synchronization execution mode
 *   IPEX_TILE_AS_DEVICE:
 *      Default = 1 | Set 0 to disable tile partition and map per root device
 * ==========GPU==========
 *
 * EXPERIMENTAL options:
 * ==========EXP==========
 *   IPEX_SIMPLE_TRACE:
 *      Default = 0 | Set 1 to enable simple trace for all operators*
 * ==========EXP==========

 * Internal options:
 * ==========INT==========
 *   IPEX_SHOW_OPTION:
 *      Default = 0 | Set 1 to show all launch option values
 *   IPEX_XPU_ONEDNN_LAYOUT:
 *      Default = 0 | Set 1 to enable onednn specific layouts
 *   IPEX_XPU_BACKEND:
 *      Default = 0 (GPU) | Set XPU_BACKEND as global IPEX backend
 *   IPEX_COMPUTE_ENG:
 *      Default = 0 (RECOMMEND) | Set RECOMMEND to select recommended compute
 engine
 *      operators: RECOMMEND, BASIC, ONEDNN, ONEMKL, XETLA
 * ==========INT==========
 */

static std::mutex s_mutex;

static Settings mySettings;

Settings& Settings::I() {
  return mySettings;
}

Settings::Settings() {
#define _(name) "IPEX_" #name

#define DPCPP_INIT_ENV_VAL(name, var, etype, show)    \
  do {                                                \
    auto env = std::getenv(_(name));                  \
    if (env) {                                        \
      try {                                           \
        int _ival = std::stoi(env, 0, 10);            \
        if (_ival <= etype##_MAX && _ival >= 0) {     \
          var = static_cast<decltype(var)>(_ival);    \
        }                                             \
      } catch (...) {                                 \
        try {                                         \
          std::string _sval(env);                     \
          for (int i = 0; i <= etype##_MAX; i++) {    \
            if (_sval == etype##_STR[i]) {            \
              var = static_cast<decltype(var)>(i);    \
              break;                                  \
            }                                         \
          }                                           \
        } catch (...) {                               \
        }                                             \
      }                                               \
    }                                                 \
    if (show) {                                       \
      std::cerr << " ** " << _(name) << ": ";         \
      if (var <= etype##_MAX && var >= 0) {           \
        std::cerr << etype##_STR[var];                \
      } else {                                        \
        std::cerr << "UNKNOW";                        \
      }                                               \
      std::cerr << " (= " << var << ")" << std::endl; \
    }                                                 \
  } while (0)

  ENV_VAL show_opt = ENV_VAL::OFF;
  DPCPP_INIT_ENV_VAL(SHOW_OPTION, show_opt, ENV_VAL, false);
  if (show_opt) {
    std::cerr << std::endl
              << " *********************************************************"
              << std::endl
              << " ** The values of all available launch options for IPEX **"
              << std::endl
              << " *********************************************************"
              << std::endl;
  }

  verbose_level = VERBOSE_LEVEL::DISABLE;
  DPCPP_INIT_ENV_VAL(VERBOSE, verbose_level, VERBOSE_LEVEL, show_opt);

  xpu_backend = XPU_BACKEND::GPU;
  DPCPP_INIT_ENV_VAL(XPU_BACKEND, xpu_backend, XPU_BACKEND, show_opt);

  sync_mode_enabled = ENV_VAL::OFF;
  DPCPP_INIT_ENV_VAL(XPU_SYNC_MODE, sync_mode_enabled, ENV_VAL, show_opt);

  tile_as_device_enabled = ENV_VAL::ON;
  DPCPP_INIT_ENV_VAL(TILE_AS_DEVICE, tile_as_device_enabled, ENV_VAL, show_opt);

  onednn_layout_enabled = ENV_VAL::OFF;
  DPCPP_INIT_ENV_VAL(
      XPU_ONEDNN_LAYOUT, onednn_layout_enabled, ENV_VAL, show_opt);

  compute_eng = COMPUTE_ENG::RECOMMEND;
  DPCPP_INIT_ENV_VAL(COMPUTE_ENG, compute_eng, COMPUTE_ENG, show_opt);

  fp32_math_mode = FP32_MATH_MODE::FP32;
  DPCPP_INIT_ENV_VAL(FP32_MATH_MODE, fp32_math_mode, FP32_MATH_MODE, show_opt);

#ifdef BUILD_SIMPLE_TRACE
  simple_trace_enabled = ENV_VAL::OFF;
  DPCPP_INIT_ENV_VAL(SIMPLE_TRACE, simple_trace_enabled, ENV_VAL, show_opt);
#endif

  if (show_opt) {
    std::cerr << " *********************************************************"
              << std::endl;
  }
} // namespace dpcpp

bool Settings::has_fp64_dtype(int device_id) {
  return dpcppGetDeviceProperties(device_id)->support_fp64;
}

bool Settings::has_2d_block_array(int device_id) {
  // FIXME: No avialble query to check 2d_block_array in sycl so far.
  // Therefore, we check FP64 capability to guess the platform status.
  return has_fp64_dtype(device_id);
}

bool Settings::has_atomic64(int device_id) {
  return dpcppGetDeviceProperties(device_id)->support_atomic64;
}

int Settings::get_verbose_level() const {
  std::lock_guard<std::mutex> lock(s_mutex);
  return static_cast<int>(verbose_level);
}

bool Settings::set_verbose_level(int level) {
  std::lock_guard<std::mutex> lock(s_mutex);
  if (level >= 0 && level <= VERBOSE_LEVEL_MAX) {
    verbose_level = static_cast<VERBOSE_LEVEL>(level);
    return true;
  }
  return false;
}

XPU_BACKEND Settings::get_backend() const {
  std::lock_guard<std::mutex> lock(s_mutex);
  return xpu_backend;
}

bool Settings::set_backend(XPU_BACKEND backend) {
  std::lock_guard<std::mutex> lock(s_mutex);
  if ((backend >= XPU_BACKEND::GPU) && (backend <= XPU_BACKEND_MAX)) {
    xpu_backend = backend;
    return true;
  }
  return false;
}

COMPUTE_ENG Settings::get_compute_eng() const {
  std::lock_guard<std::mutex> lock(s_mutex);
  return compute_eng;
}

bool Settings::set_compute_eng(COMPUTE_ENG eng) {
  std::lock_guard<std::mutex> lock(s_mutex);
  if ((eng >= COMPUTE_ENG::RECOMMEND) && (eng <= COMPUTE_ENG_MAX)) {
    compute_eng = eng;
    return true;
  }
  return false;
}

bool Settings::is_sync_mode_enabled() const {
  std::lock_guard<std::mutex> lock(s_mutex);
  return sync_mode_enabled == ENV_VAL::ON;
}

void Settings::enable_sync_mode() {
  std::lock_guard<std::mutex> lock(s_mutex);
  sync_mode_enabled = ENV_VAL::ON;
}

void Settings::disable_sync_mode() {
  std::lock_guard<std::mutex> lock(s_mutex);
  sync_mode_enabled = ENV_VAL::OFF;
}

bool Settings::is_tile_as_device_enabled() const {
  std::lock_guard<std::mutex> lock(s_mutex);
  return tile_as_device_enabled == ENV_VAL::ON;
}

void Settings::enable_tile_as_device() {
  if (!dpcppIsDevPoolInit()) {
    std::lock_guard<std::mutex> lock(s_mutex);
    tile_as_device_enabled = ENV_VAL::ON;
  }
}

void Settings::disable_tile_as_device() {
  if (!dpcppIsDevPoolInit()) {
    std::lock_guard<std::mutex> lock(s_mutex);
    tile_as_device_enabled = ENV_VAL::OFF;
  }
}

bool Settings::is_onednn_layout_enabled() const {
  std::lock_guard<std::mutex> lock(s_mutex);
  return onednn_layout_enabled == ENV_VAL::ON;
}

void Settings::enable_onednn_layout() {
  std::lock_guard<std::mutex> lock(s_mutex);
  onednn_layout_enabled = ENV_VAL::ON;
}

void Settings::disable_onednn_layout() {
  std::lock_guard<std::mutex> lock(s_mutex);
  onednn_layout_enabled = ENV_VAL::OFF;
}

FP32_MATH_MODE Settings::get_fp32_math_mode() const {
  std::lock_guard<std::mutex> lock(s_mutex);
  return fp32_math_mode;
}

bool Settings::set_fp32_math_mode(FP32_MATH_MODE mode) {
  std::lock_guard<std::mutex> lock(s_mutex);
  if ((mode >= FP32_MATH_MODE::FP32) && (mode <= FP32_MATH_MODE_MAX)) {
    fp32_math_mode = mode;
    return true;
  }
  return false;
}

bool Settings::set_onednn_verbose(int level) {
  return xpu::oneDNN::set_onednn_verbose(level);
}

bool Settings::set_onemkl_verbose(int level) {
  return xpu::oneMKL::set_onemkl_verbose(level);
}

bool Settings::is_onemkl_enabled() const {
#if defined(USE_ONEMKL)
  return true;
#else
  return false;
#endif
}

bool Settings::is_multi_context_enabled() const {
#if defined(USE_MULTI_CONTEXT)
  return true;
#else
  return false;
#endif
}

bool Settings::is_channels_last_1d_enabled() const {
#if defined(USE_CHANNELS_LAST_1D)
  return true;
#else
  return false;
#endif
}

bool Settings::is_jit_quantization_save_enabled() const {
#if defined(BUILD_JIT_QUANTIZATION_SAVE)
  return true;
#else
  return false;
#endif
}

bool Settings::is_xetla_enabled() const {
#if defined(USE_XETLA)
  return true;
#else
  return false;
#endif
}

bool Settings::is_simple_trace_enabled() const {
#ifdef BUILD_SIMPLE_TRACE
  std::lock_guard<std::mutex> lock(s_mutex);
  return simple_trace_enabled == ENV_VAL::ON;
#else
  return false;
#endif
}

void Settings::enable_simple_trace() {
#ifdef BUILD_SIMPLE_TRACE
  std::lock_guard<std::mutex> lock(s_mutex);
  simple_trace_enabled = ENV_VAL::ON;
#endif
}

void Settings::disable_simple_trace() {
#ifdef BUILD_SIMPLE_TRACE
  std::lock_guard<std::mutex> lock(s_mutex);
  simple_trace_enabled = ENV_VAL::OFF;
#endif
}

} // namespace dpcpp

/* FIXME: The backend is not ready for now.
 * Do not export to public
XPU_BACKEND get_backend() {
  return dpcpp::Settings::I().get_backend();
}

bool set_backend(XPU_BACKEND backend) {
  return dpcpp::Settings::I().set_backend(backend);
}
*/

FP32_MATH_MODE get_fp32_math_mode() {
  return dpcpp::Settings::I().get_fp32_math_mode();
}

bool set_fp32_math_mode(FP32_MATH_MODE mode) {
  return dpcpp::Settings::I().set_fp32_math_mode(mode);
}

} // namespace xpu
