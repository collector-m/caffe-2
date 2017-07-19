#include <boost/thread.hpp>
#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
///http://blog.csdn.net/xizero00/article/details/50901204
///http://www.jianshu.com/p/4a8c07de0a84
namespace caffe {

/*
 * 首先介绍一下boost::weak_ptr;
   弱引用是为了解决shared_ptr在循环引用下的内存释放问题而产生的。
   弱引用当引用的对象活着的时候不一定存在。仅仅是当它存在的时候的一个引用。
   弱引用并不修改该对象的引用计数，这意味这弱引用它并不对对象的内存进行管理，
   在功能上类似于普通指针，然而一个比较大的区别是，弱引用能检测到所管理的对象是否已经被释放，
   从而避免访问非法内存。由于弱引用不更改引用计数，类似普通指针，只要把循环引用的一方使用弱引用，即可解除循环引用 */

using boost::weak_ptr;/// 用于解决share_ptr在循环引用的时候的内存释放

map<const string, weak_ptr<DataReader::Body> > DataReader::bodies_;
static boost::mutex bodies_mutex_;
/*从共享的资源读取数据然后排队输入到数据层，每个资源创建单个线程，即便是使用多个GPU在并行任务中求解。这就保证对于频繁
 * 读取数据库，并且每个求解的线程使用的子数据是不同的。数据成功设计就是这样使在求解时数据保持一种循环地并行训练。
总结：实际上该数据层就是调用了封装层的DB来读取数据，此外还简单封装了boost的线程库，然后自己封装了个阻塞队列。
构造函数，传入的是网络的参数,初始化queue_pair_（里面包含两个阻塞队列free_和full_）*/
DataReader::DataReader(const LayerParameter& param)
    : queue_pair_(new QueuePair(  //创建一个QueuePair以初始化queue_pair_变量
        param.data_param().prefetch() * param.data_param().batch_size())) {
  // Get or create a body
  boost::mutex::scoped_lock lock(bodies_mutex_);/// 首先创建或者获取一个body实例
  string key = source_key(param);/// 从网络参数中获取key
  weak_ptr<Body>& weak = bodies_[key];/// bodies_:map,存放string到Body指针的映射
  body_ = weak.lock();
  if (!body_) {/// 如果bodies是空的
    body_.reset(new Body(param));/// 则新建Body实例到body_
    bodies_[key] = weak_ptr<Body>(body_);///然后存放到bodies_中去
  }
  body_->new_queue_pairs_.push(queue_pair_);/// 并将queue_pair放入body_中的new_queue_pairs_中去
}

DataReader::~DataReader() {
  string key = source_key(body_->param_);
  body_.reset();
  boost::mutex::scoped_lock lock(bodies_mutex_);/// 上锁
  if (bodies_[key].expired()) {
    bodies_.erase(key);/// map里面的erase
  }
}

//

DataReader::QueuePair::QueuePair(int size) {
  ///根据给定的size初始化的若干个Datum的实例到free_里面。
  // Initialize the free queue with requested number of datums
  /// 一开始全部压入free_
  for (int i = 0; i < size; ++i) {
    free_.push(new Datum());///未经过初始化。
  }
}

DataReader::QueuePair::~QueuePair() {
  ///就是将full_和free_这两个队列里面的Datum对象全部delete。
  Datum* datum;
  while (free_.try_pop(&datum)) {
    delete datum;
  }
  while (full_.try_pop(&datum)) {
    delete datum;
  }
}


/* Body继承于InternalThread*/

DataReader::Body::Body(const LayerParameter& param)///Body类的构造函数，实际上是给定网络的参数，然后开始启动内部线程
    : param_(param),
      new_queue_pairs_() {
  StartInternalThread();
  /// 调用InternalThread内部的函数来初始化运行环境以及新建线程去执行虚函数InternalThreadEntry的内容
}

DataReader::Body::~Body() {
  /// 析构，停止线程
  StopInternalThread();
}
///复写父类的函数：自身线程需要执行的函数，首先打开数据库，然后设置游标，然后设置QueuePair指针容器
    void DataReader::Body::InternalThreadEntry() {
  shared_ptr<db::DB> db(db::GetDB(param_.data_param().backend()));/// 获取所给定的数据源的类型来得到DB的指针
  db->Open(param_.data_param().source(), db::READ);/// 从网络参数中给定的DB的位置打开DB
  shared_ptr<db::Cursor> cursor(db->NewCursor());/// 新建游标指针
  vector<shared_ptr<QueuePair> > qps;/// 新建QueuePair指针容器，QueuePair里面包含了free_和full_这两个阻塞队列
  try {
    int solver_count = param_.phase() == TRAIN ? Caffe::solver_count() : 1;/// 根据网络参数的阶段来设置solver_count

    // To ensure deterministic runs, only start running once all solvers
    // are ready. But solvers need to peek on one item during initialization,
    // so read one item, then wait for the next solver.
    for (int i = 0; i < solver_count; ++i) {
      shared_ptr<QueuePair> qp(new_queue_pairs_.pop());
      read_one(cursor.get(), qp.get());/// 读取一个数据
      qps.push_back(qp);/// 压入
    }
    // Main loop
    while (!must_stop()) {
      for (int i = 0; i < solver_count; ++i) {
        read_one(cursor.get(), qps[i].get());
      }
      // Check no additional readers have been created. This can happen if
      // more than one net is trained at a time per process, whether single
      // or multi solver. It might also happen if two data layers have same
      // name and same source.
      CHECK_EQ(new_queue_pairs_.size(), 0);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
}

/// 从数据库中获取一个数据
void DataReader::Body::read_one(db::Cursor* cursor, QueuePair* qp) {
  Datum* datum = qp->free_.pop();/// 从QueuePair中的free_队列pop出一个
  // TODO deserialize in-place instead of copy?
  datum->ParseFromString(cursor->value());/// 然后解析cursor中的值
  qp->full_.push(datum);/// 然后压入QueuePair中的full_队列

  // go to the next iter
  cursor->Next();/// 游标指向下一个
  if (!cursor->valid()) {
    DLOG(INFO) << "Restarting data prefetching from start.";
    cursor->SeekToFirst();/// 如果游标指向的位置已经无效了则指向第一个位置
  }
}

}  // namespace caffe
