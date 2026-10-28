// Host-only extension of the archived32-line runtime query example.
// No kernel definitions, calls, allocations, launch, device selection or setters.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <cstdlib>
#include <cstdint>
#include <sstream>
#include <string>

static std::string quoted(const char* text) {
    std::ostringstream s; s << '"';
    for (const unsigned char c : std::string(text ? text : "")) {
        if (c == '\\' || c == '"') s << '\\' << c;
        else if (c < 32) { const char* hex = "0123456789abcdef"; s << "\\u00" << hex[c >> 4] << hex[c & 15]; }
        else s << c;
    }
    s << '"'; return s.str();
}

extern "C" const char* cjj_native_capacity_query(
        const char* library, const char* symbol, unsigned long long expected_offset,
        int requested_device) {
    static std::string result;
    auto unavailable = [&](const char* phase, const char* reason) {
        result = "{\"status\":\"diagnostic_unavailable\",\"binding_verified\":false,\"phase\":" +
                 quoted(phase) + ",\"reason\":" + quoted(reason) + "}";
        return result.c_str();
    };
    // Torch must already have loaded the exact SO normally. This cannot load a new SO.
    void* handle = dlopen(library, RTLD_NOW | RTLD_NOLOAD);
    if (!handle) return unavailable("RTLD_NOLOAD", dlerror());
    dlerror(); void* stub = dlsym(handle, symbol); const char* err = dlerror();
    if (err || !stub) return unavailable("dlsym", err ? err : "null stub");
    Dl_info info{};
    if (!dladdr(stub, &info) || !info.dli_fname || !info.dli_sname || !info.dli_saddr)
        return unavailable("dladdr", "missing loaded-object provenance");
    char* expected_path = realpath(library, nullptr); char* loaded_path = realpath(info.dli_fname, nullptr);
    const bool path_matches = expected_path && loaded_path && std::string(expected_path) == loaded_path;
    free(expected_path); free(loaded_path);
    const auto offset = reinterpret_cast<std::uintptr_t>(stub) - reinterpret_cast<std::uintptr_t>(info.dli_fbase);
    if (!path_matches || std::string(info.dli_sname) != symbol || info.dli_saddr != stub || offset != expected_offset)
        return unavailable("dladdr_identity", "wrong SO, export, exact symbol start or virtual offset");
    int device = -1, runtime_version = -1, driver_version = -1;
    const auto dr = cudaGetDevice(&device);
    cudaDeviceProp properties{};
    const auto pr = dr == cudaSuccess ? cudaGetDeviceProperties(&properties, device) : dr;
    const auto rr = cudaRuntimeGetVersion(&runtime_version);
    const auto vr = cudaDriverGetVersion(&driver_version);
    std::ostringstream s;
    s << "{\"binding_verified\":true,\"loaded_path\":" << quoted(info.dli_fname)
      << ",\"registered_host_stub\":" << quoted(info.dli_sname) << ",\"stub_offset\":" << offset
      << ",\"device_returncode\":" << int(dr) << ",\"properties_returncode\":" << int(pr)
      << ",\"device\":" << device << ",\"runtime_version_returncode\":" << int(rr)
      << ",\"runtime_version\":" << runtime_version << ",\"driver_version_returncode\":" << int(vr)
      << ",\"driver_version\":" << driver_version << ",\"device_properties\":";
    if (pr == cudaSuccess) {
        s << "{\"name\":" << quoted(properties.name) << ",\"major\":" << properties.major
          << ",\"minor\":" << properties.minor << ",\"warpSize\":" << properties.warpSize
          << ",\"multiProcessorCount\":" << properties.multiProcessorCount
          << ",\"maxThreadsPerBlock\":" << properties.maxThreadsPerBlock
          << ",\"maxThreadsPerMultiProcessor\":" << properties.maxThreadsPerMultiProcessor
          << ",\"sharedMemPerBlock\":" << properties.sharedMemPerBlock
          << ",\"sharedMemPerMultiprocessor\":" << properties.sharedMemPerMultiprocessor
          << ",\"regsPerBlock\":" << properties.regsPerBlock
          << ",\"regsPerMultiprocessor\":" << properties.regsPerMultiprocessor
          << ",\"totalGlobalMem\":" << properties.totalGlobalMem << "}";
    } else s << "null";
    if (dr != cudaSuccess || pr != cudaSuccess || device != requested_device) {
        s << ",\"status\":\"diagnostic_unavailable\",\"reason\":\"device query failed or current device differs; no device switch\",\"runtime_capacity_queried\":false}";
        result = s.str(); return result.c_str();
    }
    cudaFuncAttributes attributes{}; int blocks = -1;
    const auto ar = cudaFuncGetAttributes(&attributes, stub);
    const auto oc = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, stub, 64, 0);
    s << ",\"runtime_capacity_queried\":true,\"threads\":64,\"dynamic_shared_bytes\":0"
      << ",\"attributes_returncode\":" << int(ar) << ",\"occupancy_returncode\":" << int(oc)
      << ",\"attributes_error\":" << quoted(cudaGetErrorString(ar))
      << ",\"occupancy_error\":" << quoted(cudaGetErrorString(oc)) << ",\"attributes\":";
    if (ar == cudaSuccess) s << "{\"numRegs\":" << attributes.numRegs << ",\"static_shared_bytes\":"
        << attributes.sharedSizeBytes << ",\"local_bytes\":" << attributes.localSizeBytes
        << ",\"maxThreadsPerBlock\":" << attributes.maxThreadsPerBlock << "}";
    else s << "null";
    s << ",\"runtime_active_blocks_capacity\":";
    if (oc == cudaSuccess) s << blocks; else s << "null";
    s << ",\"status\":" << quoted(ar == cudaSuccess && oc == cudaSuccess ? "queried" : "diagnostic_unavailable")
      << ",\"measured_residency\":false,\"explicit_kernel_launch_calls\":0}";
    result = s.str(); return result.c_str();
}
