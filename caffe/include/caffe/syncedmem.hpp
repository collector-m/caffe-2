#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>

#include "caffe/common.hpp"
/**
http://www.jianshu.com/p/b105578b214b
http://blog.csdn.net/u011511601/article/details/51395481
http://www.cnblogs.com/louyihang-loves-baiyan/p/5150554.html
内存分配与释放由两个(不属于SyncedMemory类)的内联函数完成:
如果是CPU模式,那么调用malloc和free来申请/释放内存,
否则调用两个全局的内联函数:CUDA的cudaMallocHost和cudaFreeHost来申请/释放显存.
*/
///负责caffe底层的内存管理.
namespace caffe {

/*
// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.

 如果机器是支持GPU的并且安装了cuda，通过cudaMallocHost分配的host memory将会被pinned，
 pinned的意思就是内存不会被paged out，我们知道内存里面是由页作为基本的管理单元。
 分配的内存可以常驻在内存空间中对效率是有帮助的，空间不会被别的进程所抢占。同样如果内存越大，
 能被分配的Pinned内存自然也越大。还有一点是，对于单一的GPU而言提升并不会太显著，
 但是对于多个GPU的并行而言可以显著提高稳定性。这里是两个封装过的函数，
 内部通过cuda来分配主机和释放内存的接口.
 */
    /// ------ 分配内存 ------
inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda) {///2级指针。
#ifndef CPU_ONLY ////GPU内存。
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaMallocHost(ptr, size));
    *use_cuda = true;
    return;
  }
#endif
  *ptr = malloc(size);///CPU内存。
  *use_cuda = false;
  CHECK(*ptr) << "host allocation of size " << size << " failed";///*ptr为空，则分配失败。
}
/// ------ 释放内存 ------CPU内存
inline void CaffeFreeHost(void* ptr, bool use_cuda) {
#ifndef CPU_ONLY
  if (use_cuda) {
    CUDA_CHECK(cudaFreeHost(ptr));
    return;
  }
#endif
  free(ptr);
}


/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 内存分配和Caffe的底层数据的切换（cpu模式和gpu模式），需要用到内存同步模块。
 * TODO(dox): more thorough description.
 */
class SyncedMemory {
 public:
    /**
第一个为简单初始化，第二个只是把 size(大小)设置了，并未申请内存。析构函数,主要就是释放数据。
own_gpu_data和own_cpu_data表示是否拥有该数据，也即在cpu或gpu中是否有其他指针指向该数据。
*/
    SyncedMemory()
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}
  ~SyncedMemory();


/**
分别是获取cpu，gpu中数据的指针，需要说明的一点是，该过程会同步数据。
有获取，就有设置，下面两个函数就是设置数据了。
 这里设置后就不是拥有该数据，即own_cpu_data或own_gpu_data就为false，因为还有data指向该数据。
一般来说，只有当同步后才会为true。也即to_cpu()或者to_gpu()后。
*/
  const void* cpu_data();///获取cpu上的数据，返回void * 指针.指针所指向的值不可修改

///用一个void * 指针修改指针,功能：先清空CPU的数据，再设置cpu_ptr_指向void*data指针
  void set_cpu_data(void* data);

  const void* gpu_data();///获取gpu数据，返回void * 指针.指针所指向的值不可修改
  void set_gpu_data(void* data);

///获取可以更改cpu数据的指针(返回void * 指针),并改变数据的状态为HEAD_AT_CPU，mutable:可读写性的
  void* mutable_cpu_data();

///获取可以更改gpu数据的指针(返回void * 指针),并改变数据的状态为HEAD_AT_GPU
  void* mutable_gpu_data();
/*
 关于SymceHead，有四种状态，分别是未初始化，数据在cpu中，数据在gpu中， 数据在cpu和gpu中都有
 */
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  SyncedHead head() { return head_; }///获得枚举值
  size_t size() { return size_; }///获得当前存储空间的数据大小

#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif

 private:
/**
功能：把数据放到cpu上
1.数据未初始化，则在cpu申请内存（申请为0）。此时状态为HEAD_AT_CPU
2.数据本来在gpu，则从gpu拷贝内存到cpu。此时状态为SYNCED
3.数据本来在cpu，不做处理
4.数据在cpu和gpu都有，不做处理
*/
void to_cpu();


/**
功能：把数据放到gpu上
1.数据未初始化，在gpu申请内存（申请为0）。此时状态为HEAD_AT_GPU
2.数据在cpu，从cpu拷贝到gpu。此时状态为SYNCED
3.数据在gpu，不做操作。
4.数据在cpu和gpu都有，不做操作。
*/
void to_gpu();
void* cpu_ptr_;///指向cpu的指针
void* gpu_ptr_;///指向gpu的指针
size_t size_;
SyncedHead head_;///表示数据存放的位置，枚举值:也表示数据最后一次修改的地点。
bool own_cpu_data_;///cpus上是否有数据
bool cpu_malloc_use_cuda_;
bool own_gpu_data_;
int gpu_device_;

DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
