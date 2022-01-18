#ifndef CAFFE_DATA_AUGMENT_HPP
#define CAFFE_DATA_AUGMENT_HPP

#include <string>
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class DataAugmenter {
 public:
  
  explicit DataAugmenter(const TransformationParameter& param);
  virtual ~DataAugmenter() {}
  
  void InitRand();
  
  int Rand(int n);
  
  void Transform(cv::Mat& cv_img);
  
  void Color(cv::Mat& cv_img);
  
  void Contrast(cv::Mat& cv_img);
  
  void Brightness(cv::Mat& cv_img);
  
  void Rotation(cv::Mat& cv_img , int);
  

 protected:

  TransformationParameter param_;
  shared_ptr<Caffe::RNG> rng_;

  bool m_has_rotation;
  bool m_has_brightness;
  bool m_has_color;
  bool m_has_contrast;

  int m_img_index;
  bool m_show_info; 
  string m_save_dir;
  };
}
#endif  // CAFFE_DATA_AUGMENT_HPP_