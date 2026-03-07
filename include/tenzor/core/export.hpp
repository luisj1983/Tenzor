#pragma once

// TENZOR_API: Mark symbols for DLL export/import on Windows, default visibility on Unix.
// When building Tenzor as a shared library, define TENZOR_BUILDING_DLL.
#if defined(_WIN32) || defined(_MSC_VER)
  #ifdef TENZOR_BUILDING_DLL
    #define TENZOR_API __declspec(dllexport)
  #else
    #define TENZOR_API __declspec(dllimport)
  #endif
#elif defined(__GNUC__) || defined(__clang__)
  #define TENZOR_API __attribute__((visibility("default")))
#else
  #define TENZOR_API
#endif
