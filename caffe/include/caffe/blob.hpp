#ifndef CAFFE_BLOB_HPP_
#define CAFFE_BLOB_HPP_

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"

///blob最大维数目
const int kMaxBlobAxes = 32;

namespace caffe {

/**
 * @brief A wrapper around SyncedMemory holders serving as the basic
 *        computational unit through which Layer%s, Net%s, and Solver%s
 *        interact.
 *
 * BLOB是SyncedMemory的包裹器,实现上是对SyncedMemory 进行了一层封装。
 * BLOB (binary large object)，二进制大对象，是一个可以存储二进制文件的容器。
 * 在计算机中，BLOB常常是数据库中用来存储二进制文件的字段类型。
 *
 * BLOB是一个大文件，典型的BLOB是一张图片或一个声音文件，由于它们的尺寸，必须使用特殊的方式来处理
 * 例如：上传、下载或者存放到一个数据库）。
 * 
 * caffe网络各层之间的数据是通过Blob来传递的,Blob是基本的计算单元
 *
 * 实际上BLOB包含了三类数据
（1）data，前向传播所用到的数据
（2）diff，反向传播所用到的数据
（3）shape，解释data和diff的shape数据
 其中data_ 和 diff_ 是指向SyncedMemory类的shared_ptr
 * TODO(dox): more thorough description.
 */
template <typename Dtype>
class Blob {
 public:
  Blob()
       : data_(), diff_(), count_(0), capacity_(0) {}

  explicit Blob(const int num, const int channels, const int height,
      const int width);
  explicit Blob(const vector<int>& shape);
  ///Reshape函数将num,channels,height,width传递给vector shape_
  void Reshape(const int num, const int channels, const int height,
      const int width);

  /**
   * @brief Change the dimensions of the blob, allocating new memory if
   *        necessary.
   *
   * This function can be called both to create an initial allocation
   * of memory, and to adjust the dimensions of a top blob during Layer::Reshape
   * or Layer::Forward. When changing the size of blob, memory will only be
   * reallocated if sufficient memory does not already exist, and excess memory
   * will never be freed.
   *
   * Note that reshaping an input blob and immediately calling Net::Backward is
   * an error; either Net::Forward or Net::Reshape need to be called to
   * propagate the new input shape to higher layers.
   *
   *
 *Blob作为一个最基础的类，其中构造函数开辟一个内存空间来存储数据，Reshape函数在Layer中的
 *reshape或者forward 操作中来adjust the dimensions of a top blob。同时在改变Blob大小时，
 *如果内存大小不够了,内存将会被重新分配，并且多余的内存不会被释放。对input的blob进行reshape,
 *如果立马调用Net::Backward是会出错的，因为reshape之后，要么Net::forward或者Net::Reshape就会
 *被调用来将新的input shape 传播到高层
 */
  ///根据shape来初始化shape_和shape_data_，以及为data_ 和diff_ 分配空间。
  void Reshape(const vector<int>& shape);
  void Reshape(const BlobShape& shape);
  void ReshapeLike(const Blob& other);

    ///返回shape_的string形式，用于打印blob的log
  inline string shape_string() const {
    ostringstream stream;
    for (int i = 0; i < shape_.size(); ++i) {
      stream << shape_[i] << " ";
    }
    stream << "(" << count_ << ")";
    return stream.str();
  }

    ///获取shape_
  inline const vector<int>& shape() const { return shape_; }


  /**
   * @brief Returns the dimension of the index-th axis (or the negative index-th
   *        axis from the end, if index is negative).
   *
   * @param index the axis index, which may be negative as it will be
   *        "canonicalized" using CanonicalAxisIndex.
   *        Dies on out of range index.
   */
  ///获取index维的大小，返回某一维的尺寸
  inline int shape(int index) const {
    return shape_[CanonicalAxisIndex(index)];
  }

  ///维数
  inline int num_axes() const { return shape_.size(); }

  /// 返回数据的所有维度的相乘,即数据的个数
  inline int count() const { return count_; }

  /**
   * @brief Compute the volume of a slice; i.e., the product of dimensions
   *        among a range of axes.
   *
   * @param start_axis The first axis to include in the slice.
   *
   * @param end_axis The first axis to exclude from the slice.
   *
多个count()函数，主要还是为了统计Blob的容量（volume），或者是某一片（slice），
从某个axis到具体某个axis的shape乘积。

   */
  ///获取某一维到结束数据的大小。左闭右开
  inline int count(int start_axis, int end_axis) const {
    CHECK_LE(start_axis, end_axis);
    CHECK_GE(start_axis, 0);
    CHECK_GE(end_axis, 0);
    CHECK_LE(start_axis, num_axes());
    CHECK_LE(end_axis, num_axes());
    int count = 1;
    for (int i = start_axis; i < end_axis; ++i) {
      count *= shape(i);
    }
    return count;
  }
  /**
   * @brief Compute the volume of a slice spanning from a particular first
   *        axis to the final axis.
   *
   * @param start_axis The first axis to include in the slice.
   */
  inline int count(int start_axis) const {
    return count(start_axis, num_axes());
  }

  ///支持负数维度索引，负数表示从后往前，返回的是正确的维度索引（相当于将负数索引进行的转换）
  /**
   * @brief Returns the 'canonical' version of a (usually) user-specified axis,
   *        allowing for negative indexing (e.g., -1 for the last axis).
   *
   * @param axis_index the axis index.
   *        If 0 <= index < num_axes(), return index.
   *        If -num_axes <= index <= -1, return (num_axes() - (-index)),
   *        e.g., the last axis index (num_axes() - 1) if index == -1,
   *        the second to last if index == -2, etc.
   *        Dies on out of range index.
   *
   *        主要是对参数索引进行标准化，以满足要求，转换坐标轴索引[-N，N]为[0，N]
   *        -1:最后一维，-2,倒数第二维，-3,-4,....。
   */
  inline int CanonicalAxisIndex(int axis_index) const {
    CHECK_GE(axis_index, -num_axes())
        << "axis " << axis_index << " out of range for " << num_axes()
        << "-D Blob with shape " << shape_string();
    CHECK_LT(axis_index, num_axes())
        << "axis " << axis_index << " out of range for " << num_axes()
        << "-D Blob with shape " << shape_string();
    if (axis_index < 0) {
      return axis_index + num_axes();
    }
    return axis_index;
  }

  ///Blob中的4个基本变量num,channel,height,width可以直接通过shape(0),shape(1),shape(2),shape(3)来访问
  /// @brief Deprecated legacy shape accessor num: use shape(0) instead.
  inline int num() const { return LegacyShape(0); }
  /// @brief Deprecated legacy shape accessor channels: use shape(1) instead.
  inline int channels() const { return LegacyShape(1); }
  /// @brief Deprecated legacy shape accessor height: use shape(2) instead.
  inline int height() const { return LegacyShape(2); }
  /// @brief Deprecated legacy shape accessor width: use shape(3) instead.
  inline int width() const { return LegacyShape(3); }
  inline int LegacyShape(int index) const {
    CHECK_LE(num_axes(), 4)
      ///检查blob的维度个数是不是小于4，也许以前的blob只有四维，但是现在的blob应该为了通用而采用了大于四维的方法
        << "Cannot use legacy accessors on Blobs with > 4 axes.";
    CHECK_LT(index, 4);
    CHECK_GE(index, -4);
    if (index >= num_axes() || index < -num_axes()) {
      // Axis is out of range, but still in [0, 3] (or [-4, -1] for reverse
      // indexing) -- this special case simulates the one-padding used to fill
      // extraneous axes of legacy blobs.
      return 1;
    }
    return shape(index);
  }

///计算一维线性偏移量,offset计算的方式也支持两种方式，一种直接指定n,c,h,w或者放到一个vector中进行计算，
///偏移量是根据对应的n,c,h,w计算所得
  inline int offset(const int n, const int c = 0, const int h = 0,
      const int w = 0) const {
    CHECK_GE(n, 0);
    CHECK_LE(n, num());
    CHECK_GE(channels(), 0);
    CHECK_LE(c, channels());
    CHECK_GE(height(), 0);
    CHECK_LE(h, height());
    CHECK_GE(width(), 0);
    CHECK_LE(w, width());
    return ((n * channels() + c) * height() + h) * width() + w;
  }

    /// 计算一维线性偏移量,只不过参数用的是vector<int>
  inline int offset(const vector<int>& indices) const {
    CHECK_LE(indices.size(), num_axes());
    int offset = 0;
    for (int i = 0; i < num_axes(); ++i) {
      offset *= shape(i);
      if (indices.size() > i) {
        CHECK_GE(indices[i], 0);
        CHECK_LT(indices[i], shape(i));
        offset += indices[i];
      }
    }
    return offset;
  }

    ///从给定的blob进行复制，如果copy_diff=true则新的blob复制的是diff反向传播数据,
    /// 如果reshape=true则改变新blob的形状
  /**
   * @brief Copy from a source Blob.
   *
   * @param source the Blob to copy from
   * @param copy_diff if false, copy the data; if true, copy the diff
   * @param reshape if false, require this Blob to be pre-shaped to the shape
   *        of other (and die otherwise); if true, Reshape this Blob to other's
   *        shape if necessary
   */
  void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false,
      bool reshape = false);
/// 获取内存中的data数据(前向传播所用的数据)
  inline Dtype data_at(const int n, const int c, const int h,
      const int w) const {
    return cpu_data()[offset(n, c, h, w)];
  }
/// 获取内存中的diff数据(反传数据)
  inline Dtype diff_at(const int n, const int c, const int h,
      const int w) const {
    return cpu_diff()[offset(n, c, h, w)];
  }

//begin 作用同上。
  inline Dtype data_at(const vector<int>& index) const {
    return cpu_data()[offset(index)];
  }

  inline Dtype diff_at(const vector<int>& index) const {
    return cpu_diff()[offset(index)];
  }
//end

/// 返回前向传播数据的data_指针
  inline const shared_ptr<SyncedMemory>& data() const {
    CHECK(data_);
    return data_;
  }
/// 返回指向diff的智能指针。
  inline const shared_ptr<SyncedMemory>& diff() const {
    CHECK(diff_);
    return diff_;
  }

/// 这里有data和diff两类数据，而这个diff就是我们所熟知的偏差，
/// 前者主要存储前向传递的数据，而后者存储的是反向传播中的梯度
  const Dtype* cpu_data() const;///获取只读指针:指向cpu_ptr_
  void set_cpu_data(Dtype* data);///设置data_的cpu_ptr_指针

  const int* gpu_shape() const;///获取只读指针:data_的gpu指针
  const Dtype* gpu_data() const;///获取只读指针:data_的gpu指针
  const Dtype* cpu_diff() const;///获取只读指针:diff_的cpu指针
  const Dtype* gpu_diff() const;///获取只读指针:diff_的gpu指针

  Dtype* mutable_cpu_data();
        ///获取可写指针:指向cpu_ptr_cpu_ptr_见SyncedMemory的mutable_cpu_data()，mutable是可读写访问

  Dtype* mutable_gpu_data();
        ///见SyncedMemory的mutable_gpu_data();
  Dtype* mutable_cpu_diff();
  Dtype* mutable_gpu_diff();

  ///更新data_的数据,减去diff_的数据，就是合并data和diff:
  /// data=-diff_+data,Y=alpha*X+Y
/*其中用到math_functions.hpp中的函数caffe_axpy(),该函数封装了cblas_saxpy，实现的是Y=alpha*X+Y。
由此，知该函数的功能是data_=(data_-diff_)。另外，该函数只实现了对double和float型数据，
对于unsigned int和int由于该函数主要是在Net中被调用，只有Blob<float>和Blob<double>型式，
因此没有定义unsigned int和int。*/
  void Update();




/*
由BlobProto对Blob进行赋值操作。reshape代表是否允许修改shape_的大小。这里有double和float两种类型的数据
*/
  /// 从protobuf序列化文件中恢复　blob对象
  void FromProto(const BlobProto& proto, bool reshape = true);




/// 将对象序列化为protobuf文件, 将blob序列化为proto，在代码中可以看到具体的体现
  void ToProto(BlobProto* proto, bool write_diff = false) const;



/*
功能：计算前向传播data_或者diff_的L1范数
说明：对向量X求其每个元素绝对值的和。|X|
*/
  /// @brief Compute the sum of absolute values (L1 norm) of the data.
  Dtype asum_data() const;

  /// @brief Compute the sum of absolute values (L1 norm) of the diff.
  Dtype asum_diff() const;



 /*
功能：计算L2范数。
说明：向量X各元素平方的和。 int sum = X^2。
*/
  /// @brief Compute the sum of squares (L2 norm squared) of the data.
  Dtype sumsq_data() const;
  /// @brief Compute the sum of squares (L2 norm squared) of the diff.
  Dtype sumsq_diff() const;


/* 用到math_function.hpp中的caffe_scal()和 * caffe_gpu_scal()函数，
 * X = alpha*X。就是对向量X乘上一个因子。 */
  /// @brief Scale the blob data by a constant factor.
  void scale_data(Dtype scale_factor);
  /// @brief Scale the blob diff by a constant factor.
  void scale_diff(Dtype scale_factor);



  /**
   * @brief Set the data_ shared_ptr to point to the SyncedMemory holding the
   *        data_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's data_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   *
   *
   * 将data_和diff_指针指向other，实现数据的共享。
   * 同时需要注意的是这个操作会引起原来Blob里面的SyncedMemory被释放，
   * 因为shared_ptr指针调用赋值运算符时，原来所指向的对象会调用析构函数。(引用计数减为０)
   *
   */
  void ShareData(const Blob& other);
  /**
   * @brief Set the diff_ shared_ptr to point to the SyncedMemory holding the
   *        diff_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's diff_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  void ShareDiff(const Blob& other);


  bool ShapeEquals(const BlobProto& other);/// 判断形状是否相等

 protected:
  shared_ptr<SyncedMemory> data_;/// 前向传播的数据
  shared_ptr<SyncedMemory> diff_;/// 反向传播的数据
  shared_ptr<SyncedMemory> shape_data_;/// 旧的形状数据
  vector<int> shape_;/// 新的形状
  int count_;///Blob中的元素个数，也就是:个数*通道数*高度*宽度 :num * channels * height * width
  int capacity_;///数据容量

  DISABLE_COPY_AND_ASSIGN(Blob);
};  // class Blob

}  // namespace caffe

#endif  // CAFFE_BLOB_HPP_
