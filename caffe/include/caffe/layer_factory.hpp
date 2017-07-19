/**
 * @brief A layer factory that allows one to register layers.
 * During runtime, registered layers can be called by passing a LayerParameter
 * protobuffer to the CreateLayer function:
 *
 *     LayerRegistry<Dtype>::CreateLayer(param);
 *
 * There are two ways to register a layer. Assuming that we have a layer like:
 *
 *   template <typename Dtype>
 *   class MyAwesomeLayer : public Layer<Dtype> {
 *     // your implementations
 *   };
 *
 * and its type is its C++ class name, but without the "Layer" at the end
 * ("MyAwesomeLayer" -> "MyAwesome").
 *
 * If the layer is going to be created simply by its constructor, in your c++
 * file, add the following line:
 *
 *    REGISTER_LAYER_CLASS(MyAwesome);
 *
 * Or, if the layer is going to be created by another creator function, in the
 * format of:
 *
 *    template <typename Dtype>
 *    Layer<Dtype*> GetMyAwesomeLayer(const LayerParameter& param) {
 *      // your implementation
 *    }
 *
 * (for example, when your layer has multiple backends, see GetConvolutionLayer
 * for a use case), then you can register the creator function instead, like
 *
 * REGISTER_LAYER_CREATOR(MyAwesome, GetMyAwesomeLayer)
 *
 * Note that each layer type should only be registered once.
 * 应该先学习工厂模式 http://blog.csdn.net/langb2014/article/details/50991315
 *
 * 再次总结: 创建一个新layer后, 先写一个静态函数创建并返回该函数的对象 (Creator),
 * 然后创建对应的LayerRegisterer对象, 该对象在构造时会调用 LayerRegistry 中的 AddCreator,
 * 将该layer 注册到 registy中去.

 */

#ifndef CAFFE_LAYER_FACTORY_H_
#define CAFFE_LAYER_FACTORY_H_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Layer;

/* LayerResistry的功能很简单，就是将类和对应的字符串类型放入到一个map当中去，
   以便灵活调用。主要就是注册类的功能,注意：每一个Layer type 只允许注册一次*/
template <typename Dtype>
class LayerRegistry {
 public:
  typedef shared_ptr<Layer<Dtype> > (*Creator)(const LayerParameter&);
  typedef std::map<string, Creator> CreatorRegistry;

  static CreatorRegistry& Registry() {
    static CreatorRegistry* g_registry_ = new CreatorRegistry();///初始化一个空的map:(caffe运行期内，只有一次)。
    return *g_registry_;///static局部变量只被初始化一次，下一次依据上一次的结果值返回
  }

  // Adds a creator.AddCreator函数用来向Registry列表中添加一组<layername, creatorhandlr>
  ///在LayerRegistry的registry list中, 添加一个layer的creator
  ///CHECK宏当条件不成立的时候，程序会中止，并且记录对应的日志信息。
  static void AddCreator(const string& type, Creator creator) {
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(type), 0)///已存在则报错
        << "Layer type " << type << " already registered.";
    registry[type] = creator;///不存在则添加
  }

  // Get a layer using a LayerParameter.CreateLayer用于根据输入的LayerParam,获取当前Layer的layername,
  // 再去registry里通过layername获取对应的creator来创建layer
  static shared_ptr<Layer<Dtype> > CreateLayer(const LayerParameter& param) {
    if (Caffe::root_solver()) {///根节点层
      LOG(INFO) << "Creating layer " << param.name();
    }
    const string& type = param.type();
    CreatorRegistry& registry = Registry();//返回 map<string, Creator> *g_registry_
    CHECK_EQ(registry.count(type), 1) << "Unknown layer type: " << type
        << " (known types: " << LayerTypeListString() << ")";
    return registry[type](param);///registry[type]即是*Creator。 (*Creator)(const LayerParameter&)
  }

  static vector<string> LayerTypeList() {///列表化
    CreatorRegistry& registry = Registry();
    vector<string> layer_types;
/* 对于用于模板定义的依赖于模板参数的名称，只有在实例化的参数中存在这个类型名，
 * 或者这个名称前使用了typename关键字来修饰，编译器才会将该名称当成是类型。
 * 如果你想直接告诉编译器T::iterator是类型而不是变量，只需用typename修饰：
 * */
    for (typename CreatorRegistry::iterator iter = registry.begin();
         iter != registry.end(); ++iter) {
      layer_types.push_back(iter->first);///map<string, Creator>
    }
    return layer_types;
  }

 private:
  /// Layer registry should never be instantiated - everything is done with its static variables. 无法实例化
  LayerRegistry() {}

  static string LayerTypeListString() {///字符串化
    vector<string> layer_types = LayerTypeList();
    string layer_types_str;
    for (vector<string>::iterator iter = layer_types.begin();
         iter != layer_types.end(); ++iter) {
      if (iter != layer_types.begin()) {
        layer_types_str += ", ";
      }
      layer_types_str += *iter;
    }
    return layer_types_str;
  }
};


template <typename Dtype>
class LayerRegisterer {
/// LayerRegistry里用map数据结构, 维护一个CreatorRegistry list, 保存各个layer的creator的函数句柄:key 是类名, val 是对应的creator函数句柄.
///这个类只有一个方法, 即其构造函数. 构造函数只做一件事: 在LayerRegistry的registry list中, 添加一个layer的creator
 public:
  LayerRegisterer(const string& type,
                  shared_ptr<Layer<Dtype> > (*creator)(const LayerParameter&)) {
    // LOG(INFO) << "Registering layer type: " << type;
    LayerRegistry<Dtype>::AddCreator(type, creator);///静态成员函数，不用实例化对象即可调用。
  }
};

//为了方便作者还弄了个宏便于注册自己写的层类
// 生成g_creator_f_type(type, creator<Dtype>)的两个函数 （double和float类型）

///在宏定义中, #是把参数字符串化，##是连接两个参数成为一个整体
///以EuclideanLossLayer为例, 在该类的最后, 调用 REGISTER_LAYER_CLASS(EuclideanLoss);来注册这一个类.
#define REGISTER_LAYER_CREATOR(type, creator)                                  \
  static LayerRegisterer<float> g_creator_f_##type(#type, creator<float>);     \
  static LayerRegisterer<double> g_creator_d_##type(#type, creator<double>)    \

/* 注册自己定义的类，类名为type，
 假设比如type=bias，那么生成如下的代码
 下面的函数直接调用你自己的类的构造函数生成一个类的实例并返回
 CreatorbiasLayer(const LayerParameter& param)
 下面的语句是为你自己的类定义了LayerRegisterer<float>类型的静态变量g_creator_f_biasLayer（float类型，实际上就是把你自己的类的字符串类型和类的实例绑定到注册表）
 static LayerRegisterer<float> g_creator_f_biasLayer(bias, CreatorbiasLayer)
 下面的语句为你自己的类定义了LayerRegisterer<double>类型的静态变量g_creator_d_biasLayer（double类型，实际上就是把你自己的类的字符串类型和类的实例绑定到注册表）
 static LayerRegisterer<double> g_creator_d_biasLayer(bias, CreatorbiasLayer)
*/
/// REGISTER_LAYER_CLASS实际上是为每一个layer创建一个creator函数.
#define REGISTER_LAYER_CLASS(type)                                             \
  template <typename Dtype>                                                    \
  shared_ptr<Layer<Dtype> > Creator_##type##Layer(const LayerParameter& param) \
  {                                                                            \
    return shared_ptr<Layer<Dtype> >(new type##Layer<Dtype>(param));           \
  }                                                                            \
  REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)

}  // namespace caffe

#endif  // CAFFE_LAYER_FACTORY_H_
