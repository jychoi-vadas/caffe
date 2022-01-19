#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>


#include "caffe/util/data_augmenter.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
DataAugmenter<Dtype>::DataAugmenter(const TransformationParameter& param)
    : param_(param) {
    InitRand();
    m_img_index = 0;

    m_has_brightness = param_.brightness() > 0 && Rand(2);
    m_has_color = param_.color() > 0 && Rand(2);
    m_has_contrast = param_.contrast() > 0 && Rand(2);
    m_has_rotation = param_.rotation() > 0 && Rand(2);
    m_has_translation = param_.translation() > 0 && Rand(2);
    m_has_padding = param_.padding() > 0 && Rand(2);

    m_show_info = param_.show_info();
    m_save_dir  = param_.save_dir();
}

template <typename Dtype>
void DataAugmenter<Dtype>::InitRand() {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
}

template <typename Dtype>
int DataAugmenter<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

template <typename Dtype>
void DataAugmenter<Dtype>::Transform(cv::Mat& cv_img) {

  if (m_save_dir.length() > 2) {
    char im_path[256];
    sprintf(im_path, "%s/%d_ori.jpg", m_save_dir.c_str(), ++m_img_index);
    cv::imwrite(im_path, cv_img);
  } 

  if (m_has_color) { 
    Color(cv_img);
  }
  
  if (m_has_contrast) { 
    Contrast(cv_img);
  }
  
  if (m_has_brightness){ 
    Brightness(cv_img); 
  }
  
  if (m_has_rotation) { 
    Rotation(cv_img, param_.rotation()); 
  }

  if (m_save_dir.length() > 2) {
    char im_path[256];
    sprintf(im_path, "%s/%d_aug.jpg", m_save_dir.c_str(), m_img_index);
    cv::imwrite( im_path, cv_img);
  }
}


template <typename Dtype>
void DataAugmenter<Dtype>::Color(cv::Mat& cv_img) {

  //alpha 0.8 - 1.2
  double alpha = (Rand(5) + 8) / 10.0;
  cv::Mat gray_image = cv_img.clone();
  cv::Mat degenerate;

  cv::cvtColor(cv_img, gray_image, CV_BGR2GRAY);
  cv::cvtColor(gray_image, degenerate, CV_GRAY2BGR);

  cv::addWeighted(degenerate, 1-alpha, cv_img, alpha, 0.0, cv_img);

  if (m_show_info) {
    LOG(INFO) << "* Apply Color: " << alpha;
  }
}

template <typename Dtype>
void DataAugmenter<Dtype>::Brightness(cv::Mat& cv_img) {

  double alpha = (Rand(5) + 8) / 10.0;
  cv::Mat zero_img = cv_img.clone();
  zero_img.setTo(cv::Scalar(0, 0, 0));
  cv::addWeighted(zero_img, 1-alpha, cv_img, alpha, 0.0, cv_img);

  if (m_show_info) {
    LOG(INFO) << "* Alpha for Brightness : " << alpha;
  }
}

template <typename Dtype>
void DataAugmenter<Dtype>::Contrast(cv::Mat& cv_img) {

  double alpha = (Rand(5) + 8) / 10.0;
  cv::Mat gray_image;
  cv::Mat degenerate;
  cv::cvtColor(cv_img, gray_image, CV_BGR2GRAY);
  gray_image.setTo(cv::mean(gray_image));
  cvtColor(gray_image, degenerate, CV_GRAY2BGR);
  cv::addWeighted(degenerate, 1-alpha, cv_img, alpha, 0.0, cv_img);

  if (m_show_info) {
    LOG(INFO) << "* Alpha for Contrast: " << alpha;
  }
}

template <typename Dtype>
void DataAugmenter<Dtype>::Rotation(cv::Mat& cv_img, const int degree) {
  double sign = (Rand(2) % 2 ? 1.0 : -1.0);
  double sign_degree = sign * (double)degree;

  cv::Mat rotated_img = cv::Mat::zeros(cv_img.cols, cv_img.rows, cv_img.type());
  cv::Point2f center(cv_img.cols / 2., cv_img.rows / 2.);    

  cv::Mat rotate_matrix = getRotationMatrix2D(center, sign_degree, 1.0);
  cv::warpAffine(cv_img, rotated_img, rotate_matrix, cv::Size(cv_img.cols, cv_img.rows));

  rotated_img.copyTo(cv_img);

  if (m_show_info) {
    LOG(INFO) << "* Degree for Rotation : " << sign_degree;
  }
}

template <typename Dtype>
void DataAugmenter<Dtype>::Translate(cv::Mat& cv_img, const int pixel) {
  float sign_x = (Rand(2) % 2 ? 1.f : -1.f);
  float sign_y = (Rand(2) % 2 ? 1.f : -1.f);
  
  float tx = sign_x * float(Rand(pixel));
  float ty = sign_y * float(Rand(pixel));
  float translation_value[] = { 1.0, 0.0, tx, 
                                0.0, 1.0, ty };
  cv::Mat translation_matrix = cv::Mat(2, 3, CV_32F, translation_value);

  cv::Mat translated_img;
  cv::warpAffine(cv_img, translated_img, translation_matrix, cv_img.size());

  translated_img.copyTo(cv_img);

  if (m_show_info) {
    LOG(INFO) << "* Pixel for Translation : " << tx << ", " << ty;
  }
}

template <typename Dtype>
void DataAugmenter<Dtype>::Pad(cv::Mat& cv_img, const int pixel) {
  int top = Rand(pixel);
  int bottom = Rand(pixel);
  int left = Rand(pixel);
  int right = Rand(pixel);
  cv::Scalar value(Rand(255), Rand(255), Rand(255));

  cv::Mat padded_img;
  cv::copyMakeBorder(cv_img, padded_img, top, bottom, left, right, cv::BORDER_CONSTANT, value);

  padded_img.copyTo(cv_img);

  if (m_show_info) {
    LOG(INFO) << "* Pixel for Translation : " << top << ", " << bottom << ", " << left << ", " << right;
  }
}

  INSTANTIATE_CLASS(DataAugmenter);
} //namespace caffe