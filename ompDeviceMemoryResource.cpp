#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <memory_resource>
#include <string>
#include <vector>
#include <omp.h>

class OMPDeviceMemoryResource : public std::pmr::memory_resource {
public:
    OMPDeviceMemoryResource(int device_num) : device_num{device_num} {}
    OMPDeviceMemoryResource() : OMPDeviceMemoryResource(omp_get_default_device()) {}
private:
    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
        return omp_target_alloc(bytes, device_num);
    }

    void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override {
        omp_target_free(p, device_num);
    }

    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
        if(auto p = dynamic_cast<const OMPDeviceMemoryResource*>(&other); p != nullptr) {
            return (p->device_num == this->device_num);
        }
        return false;
    }

    int device_num;
};

void omp_memcpy_host_to_device(void* device_ptr, const void* host_ptr, std::size_t bytes, int device_num) {
    omp_target_memcpy(device_ptr, host_ptr, bytes, 0, 0, device_num, omp_get_initial_device());
}

void omp_memcpy_host_to_device(void* device_ptr, const void* host_ptr, std::size_t bytes) {
    omp_memcpy_host_to_device(device_ptr, host_ptr, bytes, omp_get_default_device());
}

void omp_memcpy_device_to_host(void* host_ptr, const void* device_ptr, std::size_t bytes, int device_num) {
    omp_target_memcpy(host_ptr, device_ptr, bytes, 0, 0, omp_get_initial_device(), device_num);
}

void omp_memcpy_device_to_host(void* host_ptr, const void* device_ptr, std::size_t bytes) {
    omp_memcpy_device_to_host(host_ptr, device_ptr, bytes, omp_get_default_device());
}

// ---------------------------------------------------------------------------------------------------

template<typename Ty>
void copy_via_device(std::vector<Ty>& dst, const std::vector<Ty>& src) {
    size_t N = src.size();
    dst.resize(N);
    auto host_src_ptr = src.data();
    auto host_dst_ptr = dst.data();

    OMPDeviceMemoryResource omp_device_memory_resource;

    std::pmr::polymorphic_allocator<Ty> omp_device_allocator(&omp_device_memory_resource);
    
    auto device_ptr = omp_device_allocator.allocate(N);

    omp_memcpy_host_to_device((void*)device_ptr, (void*)host_src_ptr, N * sizeof(Ty));
    omp_memcpy_device_to_host((void*)host_dst_ptr, (void*)device_ptr, N * sizeof(Ty));

    omp_device_allocator.deallocate(device_ptr, N);
}

int main() {
    std::vector<int> host_array_1(10, 2);
    std::vector<int> host_array_2(10, 0);
    assert(host_array_1 != host_array_2);

    copy_via_device(host_array_2, host_array_1);

    assert(host_array_1 == host_array_2);

    std::cout << "Copy via device was successful!\n";
    return 0;
}
