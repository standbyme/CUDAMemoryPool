#include <iostream>
#include <thread>
#include <mutex>

#include "memory_pool.h"


int main() {
    cudaStream_t stream;
    checkCuda(cudaStreamCreate(&stream));

    std::shared_ptr<MemoryPool> memoryPool = std::make_shared<MemoryPool>();
    // MemoryPool is not multi-thread safe,
    // add additional measures to prevent multiple threads from modifying the memory pool at the same time.
    std::mutex mtx;

    // Before allocating memory from the memory pool,
    // you need to register **all** the streams that may use memory space allocated from the memory pool.
    const std::lock_guard<std::mutex> lock(mtx);
    memoryPool->add_track_stream(stream);

    // Get 256 bytes cuda memory block.
    // Only when the memory pool does not have a memory block that meets the requirements,
    // a new memory block is actually allocated from the physical device.
    // Requirements:
    // - The returned memory block size should be greater than the required size.
    // - The returned memory block size should be less than twice the requested size.

    // The returned memory space is guaranteed to meet the requested size.
    // If the user reads and writes beyond the requested size, undefined behavior may occur.
    int nbytes = 256;
    void *ret = memoryPool->get(nbytes);

    std::cout << "Other code with allocated space(ret)" << std::endl;

    // Put the memory space back into the memory pool.
    memoryPool->back(ret);
    

    return 0;
}
