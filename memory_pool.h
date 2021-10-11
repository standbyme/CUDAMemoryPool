#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <algorithm>
#include <memory>
#include <vector>
#include <unordered_set>
#include <unordered_map>


inline
cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
    return result;
}

#define CHECK(x)                                           \
  if (!(x))                                                \
    std::cout << "Check failed: " #x << ": "

typedef void *StreamHandle;

class MemoryBlock {

public:
    const size_t nbytes;

    explicit MemoryBlock(const size_t &nbytes, std::shared_ptr<std::unordered_set<StreamHandle>> track_streams)
            : marked_free(true), track_streams(std::move(track_streams)), nbytes(nbytes) {
        checkCuda(cudaMalloc((void **) &ptr, nbytes));
    }

    ~MemoryBlock();

    bool isFree();

    void clearEvents();

    bool marked_free;
    std::vector<cudaEvent_t> events;
    const void *ptr = {};
    std::shared_ptr<std::unordered_set<StreamHandle>> track_streams;
};

struct Comp {
    bool operator()(std::shared_ptr<MemoryBlock> &s, const long unsigned int &i) const { return s->nbytes < i; }
};

class MemoryPool {
private:
    // sorted vector by nbytes
    std::vector<std::shared_ptr<MemoryBlock>> all_vector;
    Comp comp;
    std::unordered_map<const void *, std::shared_ptr<MemoryBlock>> all_map;
    std::shared_ptr<std::unordered_set<StreamHandle>> track_streams =
            std::make_shared<std::unordered_set<StreamHandle>>();

public:

    void *get(const size_t &min_nbytes);

    void back(void *ret);

    void add_track_stream(void *stream);
};

MemoryBlock::~MemoryBlock() {
    checkCuda(cudaFree((void *) ptr));
}

bool MemoryBlock::isFree() {
    if (marked_free) {
        auto result = std::all_of(events.begin(), events.end(),
                                  [](cudaEvent_t e) {
                                      cudaError_t error = cudaEventQuery(e);
                                      CHECK(error == cudaSuccess || error == cudaErrorNotReady) << "CUDA: "
                                                                                                << cudaGetErrorString(
                                                                                                        error);
                                      return error == cudaSuccess;
                                  });
        if (result) {
            clearEvents();
            return true;
        } else {
            return false;
        }
    } else {
        return false;
    }
}

void MemoryBlock::clearEvents() {
    for (auto &event: events) {
        checkCuda(cudaEventDestroy(event));
    }
    events.clear();
}

void MemoryPool::back(void *ret) {
    auto find = all_map.find(ret);
    CHECK(find != all_map.end());

    CHECK(!(*find).second->marked_free);

    (*find).second->marked_free = true;
}

void *MemoryPool::get(const size_t &min_nbytes) {
    auto lower_bound_iter = std::lower_bound(all_vector.begin(), all_vector.end(), min_nbytes, comp);
    if (lower_bound_iter == all_vector.end()) {
        auto memory_block = std::make_shared<MemoryBlock>(min_nbytes, track_streams);

        CHECK(memory_block->marked_free);
        CHECK(memory_block->events.empty());

        memory_block->marked_free = false;

        for (auto &stream: *memory_block->track_streams) {
            cudaEvent_t event;
            checkCuda(cudaEventCreate(&event));

            auto cuStream = static_cast<cudaStream_t>(stream);
            checkCuda(cudaEventRecord(event, cuStream));

            memory_block->events.emplace_back(event);
        }

        auto ptr = (void *) memory_block->ptr;
        all_map.insert({ptr, memory_block});
        all_vector.push_back(memory_block);

        return ptr;
    }

    auto find_if_iter = std::find_if(lower_bound_iter, all_vector.end(),
                                     [](const std::shared_ptr<MemoryBlock> &s) { return s->isFree(); });

    if (find_if_iter == all_vector.end() || (*find_if_iter)->nbytes / min_nbytes > 2) {
        auto memory_block = std::make_shared<MemoryBlock>(min_nbytes, track_streams);

        CHECK(memory_block->marked_free);
        CHECK(memory_block->events.empty());

        memory_block->marked_free = false;

        for (auto &stream: *memory_block->track_streams) {
            cudaEvent_t event;
            checkCuda(cudaEventCreate(&event));

            auto cuStream = static_cast<cudaStream_t>(stream);
            checkCuda(cudaEventRecord(event, cuStream));

            memory_block->events.emplace_back(event);
        }

        auto ptr = (void *) memory_block->ptr;
        all_map.insert({ptr, memory_block});
        all_vector.insert(lower_bound_iter, memory_block);

        return ptr;
    } else {
        CHECK((*find_if_iter)->marked_free);
        CHECK((*find_if_iter)->events.empty());

        (*find_if_iter)->marked_free = false;

        for (auto &stream: *(*find_if_iter)->track_streams) {
            cudaEvent_t event;
            checkCuda(cudaEventCreate(&event));

            auto cuStream = static_cast<cudaStream_t>(stream);
            checkCuda(cudaEventRecord(event, cuStream));

            (*find_if_iter)->events.emplace_back(event);
        }

        auto ptr = (void *) (*find_if_iter)->ptr;
        return ptr;
    }
}

void MemoryPool::add_track_stream(void *stream) {
  track_streams->emplace(stream);
}

#endif //MEMORY_POOL_H
