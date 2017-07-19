#include <boost/thread.hpp>
#include <string>

#include "caffe/data_reader.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/parallel.hpp"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {
/*
Boost.Threads支持两大类型的mutex：简单mutex和递归mutex。一个简单的mutex只能被锁住一次，
假如同一线程试图两次锁定mutex，将会产生死锁。对于递归mutex，一个线程可以多次锁定一个mutex，
但必须以同样的次数对mutex进行解锁，否则其他线程将无法锁定该mutex。
在上述两大类mutex的基础上，一个线程如何锁定一个mutex也有些不同变化。一个线程有3种可能方法来锁定mutex：
mutex类定义了内嵌的typedef来实现RAII(Resource Acquisition In Initialization，注：在初始化时资源获得)
用以对一个mutex进行加锁或者解锁，这就是所谓的Scoped Lock模式。要构建一个这种类型的锁，需要传送一个mutex引用，
构造函数将锁定mutex，析构函数将解锁mutex。C++语言规范确保了析构函数总是会被调用，所以即使有异常抛出，
mutex也会被正确地解锁。这种模式确保了mutex的正确使用。不过必须清楚，尽管Scoped Lock模式保证了mutex被正确解锁，
但它不能保证在有异常抛出的时候，所有共享资源任然处于有效的状态，所以，就像进行单线程编程一样，必须确保异常不会
让程序处于不一致的状态。同时，锁对象不能传送给另外一个线程，因为他们所维护的状态不会受到此种用法的保护。
*/

template<typename T>
class BlockingQueue<T>::sync {
 public:
  mutable boost::mutex mutex_;
  boost::condition_variable condition_;
};

template<typename T>
BlockingQueue<T>::BlockingQueue()
    : sync_(new sync()) {
}

///boost::mutex::scoped_lock:
/// http://www.boost.org/doc/libs/1_37_0/doc/html/boost/interprocess/scoped_lock.html
///http://en.cppreference.com/w/cpp/thread/scoped_lock
template<typename T>
void BlockingQueue<T>::push(const T& t) {
///首先尝试锁住，然后将数据push到队列（queue_ 是std::queue<T> 类型的），然后unlock，条件变量通知。
  boost::mutex::scoped_lock lock(sync_->mutex_);
  queue_.push(t);
  lock.unlock();
  sync_->condition_.notify_one();
/*
 condition_variable的使用要和一个判断条件绑定，在把判断条件更新为true后notify，
 wait收到notify后要判断条件是否成立 如成立结束等待，否则继续等待.
 判断条件放在一个判断谓词条件的循环里面,也可以直接向wait传入返回条件的lamda.
  关于互斥锁：
  typedef unique_lock<mutex> scoped_lock;
  scoped_lock是unique_lock<mutex>类型，因此通过查看boost的文档知道：
  std::unique_lock<std::mutex> is the tool of choice when your locking needs are more
  complex than a simple lock at the beginning followed unconditionally by an unlock at the end.
  也就是说当你的锁需求比简单的情况：一般的应用都是以lock开始，然后最后再unlock这样的情况， 但是更复杂的时候你就需要
  scoped_lock:开始需要lock，但是结束时不一定要unlock*/

/*  条件变量是提供了一种机制，该机制能够等待另一个线程发来的通知，如果另一个线程满足某个条件的话。
  通常使用条件变量是这样的，一个线程锁住mutex，然后wait，当该线程醒来的时候会检查条件变量的值是否true，
  如果是则放行，否则继续睡.*/
}

template<typename T>
bool BlockingQueue<T>::try_pop(T* t) {
  boost::mutex::scoped_lock lock(sync_->mutex_);

  if (queue_.empty()) {
    return false;
  }

  *t = queue_.front();
  queue_.pop();
  return true;
}

template<typename T>
T BlockingQueue<T>::pop(const string& log_on_wait) {
  boost::mutex::scoped_lock lock(sync_->mutex_);///加锁

  while (queue_.empty()) {
    if (!log_on_wait.empty()) { ///字符串为空
      LOG_EVERY_N(INFO, 1000)<< log_on_wait;
    }
    sync_->condition_.wait(lock);/// 如果队列一直为空则一直阻塞等待于此。
  }

  T t = queue_.front();/// 否则取出
  queue_.pop();
  return t;
}

template<typename T>
bool BlockingQueue<T>::try_peek(T* t) { ///判断队列首部是不是有数据。与peek的区别是不等待队列，立即返回。
  boost::mutex::scoped_lock lock(sync_->mutex_);

  if (queue_.empty()) {
    return false;
  }

  *t = queue_.front();
  return true;
}

template<typename T>
T BlockingQueue<T>::peek() {///peek函数取出队列首部的数据，使用条件变量来实现同步。一定会取出一个数据
  boost::mutex::scoped_lock lock(sync_->mutex_);

  while (queue_.empty()) {
    sync_->condition_.wait(lock);
  }

  return queue_.front();
}

template<typename T>
size_t BlockingQueue<T>::size() const {
  boost::mutex::scoped_lock lock(sync_->mutex_);
  return queue_.size();
}
///最后偏特化了几个类型的BlockingQueue类
template class BlockingQueue<Batch<float>*>;
template class BlockingQueue<Batch<double>*>;
template class BlockingQueue<Datum*>;
template class BlockingQueue<shared_ptr<DataReader::QueuePair> >;
template class BlockingQueue<P2PSync<float>*>;
template class BlockingQueue<P2PSync<double>*>;

}  // namespace caffe
