// 仅SAVE：SDK编译器决定CUDA→IX宏、枚举及stream类型，不手写运行库ABI。
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

struct Biv150SaveDescriptor {
    const void* source;
    void* destination;
    std::size_t source_available;
    std::size_t destination_available;
    std::size_t width;
    std::size_t height;
    std::size_t source_pitch;
    std::size_t destination_pitch;
    std::uint32_t kind;  // 1:连续线性，2:正pitch二维。
    std::uint32_t reserved;
};

static_assert(sizeof(void*) == 8 && sizeof(std::size_t) == 8, "requires 64-bit SDK ABI");
static_assert(std::is_pointer<cudaStream_t>::value, "SDK stream must be opaque pointer");
static_assert(sizeof(cudaStream_t) == sizeof(void*), "SDK stream width mismatch");
static_assert(sizeof(Biv150SaveDescriptor) == 72, "descriptor ABI mismatch");

extern "C" std::uint32_t biv150_pitched_copy_abi() { return 1; }
extern "C" std::size_t biv150_pitched_copy_descriptor_size() { return sizeof(Biv150SaveDescriptor); }

extern "C" int biv150_pitched_copy_max_pitch(int device, std::size_t* output) {
    if (!output || device != 0) return -1001;
    *output = 0;
    int value = 0;
    const auto code = cudaDeviceGetAttribute(&value, cudaDevAttrMaxPitch, device);
    if (code != cudaSuccess) return static_cast<int>(code);
    if (value <= 0) return -1002;
    *output = static_cast<std::size_t>(value);
    return 0;
}

static bool extent(std::size_t width, std::size_t height, std::size_t pitch,
                   std::size_t available, const void* pointer) {
    if (!pointer || !width || !height || pitch < width) return false;
    const auto maximum = std::numeric_limits<std::size_t>::max();
    if (height - 1 > (maximum - width) / pitch) return false;
    const std::size_t bytes = (height - 1) * pitch + width;
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    return bytes <= available && address <= maximum - bytes;
}

// 完整数组/边界/overflow先验后才提交；方向固定为真实SDK DeviceToHost。
// 指针来源及pinned/device/layout由Python实际Tensor描述符验证，不查询或修改其值。
extern "C" int biv150_pitched_save_batch(const Biv150SaveDescriptor* descriptors,
                                        std::size_t count, void* stream,
                                        std::size_t* submitted) {
    if (!submitted) return -1001;
    *submitted = 0;
    if (!descriptors || count == 0 || count > 65536) return -1001;
    for (std::size_t i = 0; i < count; ++i) {
        const auto& d = descriptors[i];
        if (d.reserved || (d.kind != 1 && d.kind != 2)
            || (d.kind == 1 && d.height != 1)
            || !extent(d.width, d.height, d.source_pitch, d.source_available, d.source)
            || !extent(d.width, d.height, d.destination_pitch, d.destination_available, d.destination))
            return -1003;
        const auto src = reinterpret_cast<std::uintptr_t>(d.source);
        const auto dst = reinterpret_cast<std::uintptr_t>(d.destination);
        const auto src_end = src + (d.height - 1) * d.source_pitch + d.width;
        const auto dst_end = dst + (d.height - 1) * d.destination_pitch + d.width;
        if (src < dst_end && dst < src_end) return -1003;
    }
    int device = -1;
    auto code = cudaGetDevice(&device);
    if (code != cudaSuccess) return static_cast<int>(code);
    if (device != 0) return -1004;
    std::size_t maximum_pitch = 0;
    const int pitch_code = biv150_pitched_copy_max_pitch(device, &maximum_pitch);
    if (pitch_code != 0) return pitch_code;
    for (std::size_t i = 0; i < count; ++i) {
        const auto& d = descriptors[i];
        if (d.kind == 2 && (d.source_pitch > maximum_pitch || d.destination_pitch > maximum_pitch))
            return -1005;
    }
    const auto sdk_stream = reinterpret_cast<cudaStream_t>(stream);
    for (std::size_t i = 0; i < count; ++i) {
        const auto& d = descriptors[i];
        code = d.kind == 1
            ? cudaMemcpyAsync(d.destination, d.source, d.width, cudaMemcpyDeviceToHost, sdk_stream)
            : cudaMemcpy2DAsync(d.destination, d.destination_pitch, d.source, d.source_pitch,
                                d.width, d.height, cudaMemcpyDeviceToHost, sdk_stream);
        if (code != cudaSuccess) return static_cast<int>(code);
        ++*submitted;
    }
    return 0;  // 仅提交成功；调用者必须同stream真event完成后才读host/释放任何引用。
}
