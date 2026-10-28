// Optional future HOST-ONLY SDK helper. Not compiled or executed by this task.
// No kernel definitions, device allocations, launches, or function-attribute writes.
// The argument must be the actual final SO's verified registered host stub,
// not a launch wrapper, device-name string, or CUfunction.
#include <cuda_runtime_api.h>
#include <cstdio>

extern "C" int cjj_query_registered_native(const void* registered_host_stub) {
    if (!registered_host_stub) return -1;
    cudaFuncAttributes attributes{};
    int blocks = -1;
    const auto ar = cudaFuncGetAttributes(&attributes, registered_host_stub);
    const auto oc = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, registered_host_stub, 64, 0);
    std::printf("{\"attributes_returncode\":%d,\"occupancy_returncode\":%d,"
                "\"threads\":64,\"dynamic_shared_bytes\":0,\"attributes\":",
                static_cast<int>(ar), static_cast<int>(oc));
    if (ar == cudaSuccess) {
        std::printf("{\"numRegs\":%d,\"static_shared_bytes\":%llu,"
                    "\"local_bytes\":%llu,\"maxThreadsPerBlock\":%d}",
                    attributes.numRegs,
                    static_cast<unsigned long long>(attributes.sharedSizeBytes),
                    static_cast<unsigned long long>(attributes.localSizeBytes),
                    attributes.maxThreadsPerBlock);
    } else std::printf("null");
    std::printf(",\"driver_active_blocks_capacity\":");
    if (oc == cudaSuccess) std::printf("%d", blocks);
    else std::printf("null");
    std::printf(",\"measured_residency\":false,\"explicit_kernel_launch_calls\":0}\n");
    std::fflush(stdout);
    return ar != cudaSuccess ? static_cast<int>(ar) : static_cast<int>(oc);
}
