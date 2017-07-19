#ifndef CAFFE_DATA_READER_HPP_
#define CAFFE_DATA_READER_HPP_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/db.hpp"
/*
http://www.jianshu.com/p/4a8c07de0a84
QueuePair与Body是DataReader的内部类。一个DataReader对应一个任务，一个Body生成一个线程
来读取数据库如examples/mnist/mnist_train_lmdb）。QueuePair为前面两者之间的衔接、通信。

一个QueuePair对应一个任务队列，从数据库（如examples/mnist/mnist_train_lmdb）中读取size个样本
BlockingQueue为一个线程安全的队列容器，其模板类型可能是Datum，Batch等。此处装的是Datum。
BlockingQueue<Datum*> free_为Datum队列，均为新new出来的，没有包含原始数据（图像）信息
BlockingQueue<Datum*> full_为从数据库读取信息后的队列，包含了原始数据（图像）信息
Datum为一个样本单元，关于Datum的定义，参见caffe.proto文件，一般来说，Datum对应于一张图像（及其label）

Body类继承了InternalThread。在构造函数了开启这个线程
Body类重载了 DataReader::Body::InternalThreadEntry()函数，从数据库读取数据的操作在该函数中实现

一个数据库只可能有Body对象，如examples/mnist/mnist_train_lmdb不管在任何线程的任何DataReader对象中，
都只会有一个Body对象，因为bodies_是静态的：所以有，一个Body的对象也可以有多个DataReader对象
此外有，一个DataReader对象可以有多个Body对象，即map<string,weak_ptr<Body>> bodies_
由代码5，6行及16行可知，每一个DataReader对应一个读的任务，即从数据库（如examples/mnist/mnist_train_lmdb）
 中读取param.data_param().prefetch() * param.data_param().batch_size()（LeNet5中默认为4×64）个样本
由此可见，一个DataReader为一个任务，通过QueuePair（也对应于该任务）“通知”Body某个数据库中读去N个样本
由代码13行可知，某个数据库（如examples/mnist/mnist_train_lmdb）对应的Body若不存在，
将新建一个Body来处理该数据库，也可以理解成新建一个唯一对应于该数据库的线程来处理该数据可。
。
read_one()从QueuePair的free_中取出一个Datum，从数据库读入数据至Datum，然后放入full_中
一个新的任务（DataReader）到来时，将把一个命令队列（QueuePair）放入到某个数据库（Body）的缓冲命令队列中（new_queue_pairs_）
9到13行从每个solver的任务中读取一个Datum，在15到18行从数据库中循环读出数据


 */

namespace caffe {

/**
 * @brief Reads data from a source to queues available to data layers.
 * A single reading thread is created per source, even if multiple solvers
 * are running in parallel, e.g. for multi-GPU training. This makes sure
 * databases are read sequentially, and that each solver accesses a different
 * subset of the database. Data is distributed to solvers in a round-robin //round-robin:循环
 * way to keep parallel training deterministic.
 */
class DataReader {///DataReader内含QueuePair和Body2个内部类。
 public:
  explicit DataReader(const LayerParameter& param);
  ~DataReader();

  inline BlockingQueue<Datum*>& free() const {
    return queue_pair_->free_;
  }
  inline BlockingQueue<Datum*>& full() const {
    return queue_pair_->full_;
  }

 protected:
  // Queue pairs are shared between a body and its readers
  ///DataReader内部的QueuePair类的实现
    class QueuePair {
   public:
    explicit QueuePair(int size);
    ~QueuePair();
///两个阻塞队列free_和full_。，该类用于在body和readers之间进行数据分享
    BlockingQueue<Datum*> free_;
    BlockingQueue<Datum*> full_;

  DISABLE_COPY_AND_ASSIGN(QueuePair);
  };

  /* A single body is created per source
Body类的实现，该类是继承自InternalThread 类,重写InternalThread内部的InternalThreadEntry函数，
此外还添加了read_one函数/内部有DataReader的友元，
以及BlockingQueue<shared_ptr<QueuePair> > new_queue_pairs_;*/
  class Body : public InternalThread {
   public:
    explicit Body(const LayerParameter& param);
    virtual ~Body();

   protected:
    void InternalThreadEntry();
    void read_one(db::Cursor* cursor, QueuePair* qp);

    const LayerParameter param_;
    BlockingQueue<shared_ptr<QueuePair> > new_queue_pairs_;

    friend class DataReader;

  DISABLE_COPY_AND_ASSIGN(Body);
  };

  // A source is uniquely identified by its layer name + path, in case
  // the same database is read from two different locations in the net.
  static inline string source_key(const LayerParameter& param) {
    return param.name() + ":" + param.data_param().source();
  }

  const shared_ptr<QueuePair> queue_pair_;
  shared_ptr<Body> body_;

  static map<const string, boost::weak_ptr<DataReader::Body> > bodies_;

DISABLE_COPY_AND_ASSIGN(DataReader);
};

}  // namespace caffe

#endif  // CAFFE_DATA_READER_HPP_
