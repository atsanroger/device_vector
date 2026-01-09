#include <openacc.h>
#include <unordered_map>
#include <mutex>
#include <cstdio>
#include <cstdlib>

#include "DeviceVector.cuh"

static std::mutex g_mu;

// 用 handle 當 key：map 住的是「這個 vector 物件」
// value 記住 host_key（acc_unmap_data 需要）
// 你採用策略 A：mapped 期間禁止 resize/free，所以 host_ptr 不會變
template <typename T>
struct MapInfo {
  void* host_key = nullptr;
};

template <typename T>
static std::unordered_map<void*, MapInfo<T>> g_map;

template <typename T>
static void do_map(void* h) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (g_map<T>.count(h)) return;

  auto* v = (GPU::IDeviceVector<T>*)h;
  void* host_key = (void*)v->host_ptr();
  void* dev_ptr  = (void*)v->device_ptr();
  size_t bytes   = v->size() * sizeof(T);

  acc_map_data(host_key, dev_ptr, bytes);
  g_map<T>[h] = {host_key};
}

template <typename T>
static void do_unmap(void* h) {
  std::lock_guard<std::mutex> lk(g_mu);
  auto it = g_map<T>.find(h);
  if (it == g_map<T>.end()) return;

  acc_unmap_data(it->second.host_key);
  g_map<T>.erase(it);
}

template <typename T>
static int is_mapped(void* h) {
  std::lock_guard<std::mutex> lk(g_mu);
  return g_map<T>.count(h) ? 1 : 0;
}

extern "C" {

int  vec_acc_is_mapped_i4(void* h){ return is_mapped<int>(h); }
int  vec_acc_is_mapped_i8(void* h){ return is_mapped<long long>(h); }
int  vec_acc_is_mapped_r4(void* h){ return is_mapped<float>(h); }
int  vec_acc_is_mapped_r8(void* h){ return is_mapped<double>(h); }

void vec_acc_map_i4(void* h){ do_map<int>(h); }
void vec_acc_map_i8(void* h){ do_map<long long>(h); }
void vec_acc_map_r4(void* h){ do_map<float>(h); }
void vec_acc_map_r8(void* h){ do_map<double>(h); }

void vec_acc_unmap_i4(void* h){ do_unmap<int>(h); }
void vec_acc_unmap_i8(void* h){ do_unmap<long long>(h); }
void vec_acc_unmap_r4(void* h){ do_unmap<float>(h); }
void vec_acc_unmap_r8(void* h){ do_unmap<double>(h); }

} // extern "C"
