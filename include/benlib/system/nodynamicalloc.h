// no_dynamic_alloc.h
#ifndef NO_DYNAMIC_ALLOC_H
#define NO_DYNAMIC_ALLOC_H

#ifdef __cplusplus
  #include <new>
  void* operator new(std::size_t) = delete;
  void* operator new[](std::size_t) = delete;
  void operator delete(void*) noexcept = delete;
  void operator delete[](void*) noexcept = delete;
#endif

#include <stdlib.h>

#undef malloc
#undef calloc
#undef realloc
#undef free

#if defined(__cplusplus)
  #define BENLIB_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
  #define BENLIB_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif

#define malloc(...)  []{ BENLIB_STATIC_ASSERT(false, "malloc is forbidden!"); return (void*)0; }()
#define calloc(...)  []{ BENLIB_STATIC_ASSERT(false, "calloc is forbidden!"); return (void*)0; }()
#define realloc(...) []{ BENLIB_STATIC_ASSERT(false, "realloc is forbidden!"); return (void*)0; }()
#define free(...)    []{ BENLIB_STATIC_ASSERT(false, "free is forbidden!"); }()

#endif // NO_DYNAMIC_ALLOC_H
