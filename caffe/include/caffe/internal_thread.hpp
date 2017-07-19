#ifndef CAFFE_INTERNAL_THREAD_HPP_
#define CAFFE_INTERNAL_THREAD_HPP_

#include "caffe/common.hpp"

/*
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX.
 */
namespace boost { class thread; }

namespace caffe {

/*
 * Virtual class encapsulate(封装) boost::thread for use in base class
 * The child class will acquire the ability to run a single thread,
 * by reimplementing the virtual function InternalThreadEntry.

 * InternalThread是一个基类，是Caffe中的多线程接口，其本质为封装了boost::thread。
 * Caffe中使用多线程的地方主要是从磁盘读取数据的地方
 * 总结一下：获取线程的状态、启动线程、以及定义的线程入口函数InternalThread::entry ，
 */
class InternalThread {
 public:
  InternalThread() : thread_() {}
  virtual ~InternalThread();

  /*
   * Caffe's thread local state will be initialized using the current
   * thread values, e.g. device id, solver index etc. The random seed
   * is initialized using caffe_rng_rand.
   * caffe的线程局部状态将会使用当前线程值来进行初始化，当前的线程的值有设备id，solver的编号、随机数种子等
   */
  void StartInternalThread();///初始化工作,实例化一个thread对象给thread_指针，该线程的执行的是entry函数.

  /* Will not return until the internal thread has exited.
   *  如果线程已经开始, 那么中断。如果没启动，则直接返回。  */
  void StopInternalThread();

  bool is_started() const;

 protected:
  /** Implement this method in your subclass  with the code you want your thread to run.
  线程真正的执行主体函数。虚函数，要求所有继承的子类实现此函数。  */
  virtual void InternalThreadEntry() {}

  /* Should be tested when running loops to exit when requested. 在当请求退出的时候应该调用该函数 */
  bool must_stop();

 private:
 /*
调用InternalThreadEntry，并且在调用之前，帮用户做好了初始化的工作:
随机数种子，CUDA、工作模式及GPU还是CPU、solver的类型。
*/
  void entry(int device, Caffe::Brew mode, int rand_seed, int solver_count,
      bool root_solver);

  shared_ptr<boost::thread> thread_;
};

}  // namespace caffe

#endif  // CAFFE_INTERNAL_THREAD_HPP_
