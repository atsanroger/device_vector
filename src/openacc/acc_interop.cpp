#include <openacc.h>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <mutex>

#include "acc_interop.h"

extern "C"
{
  int *vec_host_i4(void *h);
  long long *vec_host_i8(void *h);
  float *vec_host_r4(void *h);
  double *vec_host_r8(void *h);

  void *vec_dev_i4(void *h);
  void *vec_dev_i8(void *h);
  void *vec_dev_r4(void *h);
  void *vec_dev_r8(void *h);

  size_t vec_size_i4(void *h);
  size_t vec_size_i8(void *h);
  size_t vec_size_r4(void *h);
  size_t vec_size_r8(void *h);
}

static std::mutex g_mu;

static std::unordered_map<void *, void *> g_map_i4;
static std::unordered_map<void *, void *> g_map_i8;
static std::unordered_map<void *, void *> g_map_r4;
static std::unordered_map<void *, void *> g_map_r8;

template <typename MAP>
static int is_mapped(MAP &m, void *h)
{
  std::lock_guard<std::mutex> lk(g_mu);
  return m.count(h) ? 1 : 0;
}

template <typename MAP>
static void do_map(MAP &m, void *h, void *host_key, void *dev_ptr, size_t bytes)
{
  std::lock_guard<std::mutex> lk(g_mu);

  if (m.count(h))
    return; // already mapped
  if (!host_key || !dev_ptr)
  {
    std::fprintf(stderr, "[acc_interop] map failed: null host_key/dev_ptr\n");
    std::abort();
  }
  if (bytes == 0)
  { // size==0: still record as mapped? I'd rather no-op.
    // no-op: don't register mapping
    return;
  }

  acc_map_data(host_key, dev_ptr, bytes);
  m[h] = host_key;
}

template <typename MAP>
static void do_unmap(MAP &m, void *h)
{
  std::lock_guard<std::mutex> lk(g_mu);

  auto it = m.find(h);
  if (it == m.end())
    return; // not mapped -> no-op

  acc_unmap_data(it->second);
  m.erase(it);
}

extern "C"
{

  // ---- is_mapped ----
  int vec_acc_is_mapped_i4(void *h) { return is_mapped(g_map_i4, h); }
  int vec_acc_is_mapped_i8(void *h) { return is_mapped(g_map_i8, h); }
  int vec_acc_is_mapped_r4(void *h) { return is_mapped(g_map_r4, h); }
  int vec_acc_is_mapped_r8(void *h) { return is_mapped(g_map_r8, h); }

  // ---- map ----
  void vec_acc_map_i4(void *h)
  {
    size_t n = vec_size_i4(h);
    do_map(g_map_i4, h, (void *)vec_host_i4(h), vec_dev_i4(h), n * sizeof(int));
  }
  void vec_acc_map_i8(void *h)
  {
    size_t n = vec_size_i8(h);
    do_map(g_map_i8, h, (void *)vec_host_i8(h), vec_dev_i8(h), n * sizeof(long long));
  }
  void vec_acc_map_r4(void *h)
  {
    size_t n = vec_size_r4(h);
    do_map(g_map_r4, h, (void *)vec_host_r4(h), vec_dev_r4(h), n * sizeof(float));
  }
  void vec_acc_map_r8(void *h)
  {
    size_t n = vec_size_r8(h);
    do_map(g_map_r8, h, (void *)vec_host_r8(h), vec_dev_r8(h), n * sizeof(double));
  }

  // ---- unmap ----
  void vec_acc_unmap_i4(void *h) { do_unmap(g_map_i4, h); }
  void vec_acc_unmap_i8(void *h) { do_unmap(g_map_i8, h); }
  void vec_acc_unmap_r4(void *h) { do_unmap(g_map_r4, h); }
  void vec_acc_unmap_r8(void *h) { do_unmap(g_map_r8, h); }

} // extern "C"
