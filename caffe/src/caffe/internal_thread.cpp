#include <boost/thread.hpp>
#include <exception>

#include "caffe/internal_thread.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

InternalThread::~InternalThread() {
  StopInternalThread();
}

bool InternalThread::is_started() const { /// 首先thread_指针不能为空，然后该线程是可等待的
  return thread_ && thread_->joinable();
}

bool InternalThread::must_stop() {
  ///  if interruption has been requested for the current thread, false otherwise
  return thread_ && thread_->interruption_requested();//后者检测当前线程是否被要求中断
}

void InternalThread::StartInternalThread() { ///初始化工作
  CHECK(!is_started()) << "Threads should persist and not be restarted.";

  int device = 0;
#ifndef CPU_ONLY
  CUDA_CHECK(cudaGetDevice(&device));
#endif
  Caffe::Brew mode = Caffe::mode();
  int rand_seed = caffe_rng_rand();
  int solver_count = Caffe::solver_count();
  bool root_solver = Caffe::root_solver();
/// boost::thread一旦被构造后，就会立刻以异步的方式执行传入的函数。

  try { ///实例化一个thread对象给thread_指针，该线程的执行的是entry函数
    thread_.reset(new boost::thread(&InternalThread::entry, this, device, mode,
          rand_seed, solver_count, root_solver));
  } catch (std::exception& e) {
    LOG(FATAL) << "Thread exception: " << e.what();
  }
}

/// 线程所要执行的函数
void InternalThread::entry(int device, Caffe::Brew mode, int rand_seed,
    int solver_count, bool root_solver) {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaSetDevice(device));
#endif
  Caffe::set_mode(mode);
  Caffe::set_random_seed(rand_seed);
  Caffe::set_solver_count(solver_count);
  Caffe::set_root_solver(root_solver);

  InternalThreadEntry();
}

/** interrupt的线程可能处于阻塞睡眠状态，我们需要从外部立即唤醒它，让其检测中断请求。
所以在interrupt操作后，需要立即后接join操作。最后，还可以选择性地补上异常检测。
     * */
void InternalThread::StopInternalThread() {
  if (is_started()) {/// 如果线程已经开始, 那么中断。如果没启动，则直接返回。
    thread_->interrupt();
    try {
      thread_->join();///唤醒线程
    } catch (boost::thread_interrupted&) {//如果已经被中断，啥也不干，因为可能是之前中断的
    } catch (std::exception& e) {
      LOG(FATAL) << "Thread exception: " << e.what();/// 如果发生其他错误则记录到日志
    }
  }
}

}  // namespace caffe
