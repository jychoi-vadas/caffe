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

    m_has_blur = param_.blur() > 0 && Rand(2);
    m_has_color = param_.color() > 0 && Rand(2);
    m_has_brightness = param_.brightness() > 0 && Rand(2);
    m_has_contrast = param_.contrast() > 0 && Rand(2);
    m_has_hue = param_.hue() > 0 && Rand(2);
    m_has_saturation = param_.saturation() > 0 && Rand(2);
    m_has_rotation = param_.rotation() > 0 && Rand(2);
    m_has_translation = param_.translation() > 0 && Rand(2);
    m_has_zoom = param_.zoom() > 0 && Rand(2);

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

  if (m_has_blur) { 
    Blur(cv_img);
  }

  if (m_has_color) { 
    Color(cv_img);
  }
  
  if (m_has_brightness){ 
    Brightness(cv_img); 
  }

  if (m_has_contrast) { 
    Contrast(cv_img);
  }

//   if (m_has_hue) { 
//     Hue(cv_img, param_.hue());
//   }

//   if (m_has_saturation) { 
//     Saturate(cv_img, param_.saturation());
//   }
  
  if (m_has_zoom) { 
    Zoom(cv_img, param_.zoom()); 
  }
  
  if (m_has_translation) { 
    Translate(cv_img, param_.translation()); 
  }

  if (m_has_rotation) { 
    Rotate(cv_img, param_.rotation()); 
  }

//   if (m_save_dir.length() > 2) {
//     char im_path[256];
//     sprintf(im_path, "%s/%d_aug.jpg", m_save_dir.c_str(), m_img_index);
//     cv::imwrite(im_path, cv_img);
//   }
}


template <typename Dtype>
void DataAugmenter<Dtype>::Blur(cv::Mat& cv_img) {
  int rand_size = 3 + Rand(4);
  int kernel_size = (rand_size % 2 ? rand_size : rand_size + 1);
  
  cv::GaussianBlur(cv_img, cv_img, cv::Size(kernel_size, kernel_size), 0);

  if (m_show_info) {
    LOG(INFO) << "* Apply Blur: " << kernel_size << " x " << kernel_size;
  }
}

template <typename Dtype>
void DataAugmenter<Dtype>::Color(cv::Mat& cv_img) {
  int rand_ch = Rand(3);
  double rand_factor = ((double)(Rand(20))) / 100.0;

  cv::Mat channels_img[3];
  cv::split(cv_img, channels_img);
  channels_img[rand_ch] = channels_img[rand_ch] * (1.0 + rand_factor);
  cv::merge(channels_img, 3, cv_img);
  cv::inRange(cv_img, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255), cv_img);

  if (m_show_info) {
    LOG(INFO) << "* Alpha for Color: " << rand_factor;
  }
}

template <typename Dtype>
void DataAugmenter<Dtype>::Brightness(cv::Mat& cv_img) {
  double rand_factor = ((double)(50.0 + Rand(100))) / 100.0;

  cv::Mat zero_img = cv::Mat::zeros(cv_img.cols, cv_img.rows, cv_img.type());
  cv::addWeighted(cv_img, rand_factor, zero_img, 1.0 - rand_factor, 0.0, cv_img);
  if (m_save_dir.length() > 2) {
    char im_path[256];
    sprintf(im_path, "%s/%d_sub.jpg", m_save_dir.c_str(), m_img_index);
    cv::imwrite(im_path, cv_img);
  }

  cv::inRange(cv_img, cv::Scalar(255, 255, 255), cv::Scalar(0, 0, 0), cv_img);
  if (m_save_dir.length() > 2) {
    char im_path[256];
    sprintf(im_path, "%s/%d_aug.jpg", m_save_dir.c_str(), m_img_index);
    cv::imwrite(im_path, cv_img);
  }


  if (m_show_info) {
    LOG(INFO) << "* Alpha for Brightness : " << rand_factor;
  }
}

template <typename Dtype>
void DataAugmenter<Dtype>::Contrast(cv::Mat& cv_img) {
  double rand_factor = ((double)(50.0 + Rand(100))) / 100.0;

  cv::Mat gray_img;
  cv::cvtColor(cv_img, gray_img, CV_BGR2GRAY);

  cv::Mat mean_img;
  cvtColor(gray_img, mean_img, CV_GRAY2BGR);

  cv::addWeighted(cv_img, rand_factor, mean_img, 1.0 - rand_factor, 0.0, cv_img);
//   cv::inRange(cv_img, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255), cv_img);

  if (m_show_info) {
    LOG(INFO) << "* Alpha for Contrast: " << rand_factor << " " << cv_img.at<Vec3b>(0, 0)[0] << " " << cv_img.at<Vec3b>(100, 100)[0];
  }
}


template <typename Dtype>
void DataAugmenter<Dtype>::Hue(cv::Mat& cv_img, const float factor) {

  if (m_show_info) {
    LOG(INFO) << "* Apply Hue: " << factor;
  }
}


template <typename Dtype>
void DataAugmenter<Dtype>::Saturate(cv::Mat& cv_img, const float factor) {

  if (m_show_info) {
    LOG(INFO) << "* Apply Saturation: " << factor;
  }
}

template <typename Dtype>
void DataAugmenter<Dtype>::Rotate(cv::Mat& cv_img, const int degree) {
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
void DataAugmenter<Dtype>::Zoom(cv::Mat& cv_img, const int pixel) {
  float sign = (Rand(2) % 2 ? 1.f : -1.f);

  int origin_w = cv_img.cols;
  int origin_h = cv_img.rows;

  cv::Mat resized_img;
  int resize_w = origin_w + sign * Rand(pixel);
  int resize_h = origin_h + sign * Rand(pixel);
  cv::resize(cv_img, resized_img, cv::Size(resize_w, resize_h));

  cv::Mat zoomed_img = cv::Mat::zeros(origin_w, origin_h, cv_img.type());
  if (sign > 0) {   // zoom in
    cv::Rect roi((int)((resize_w - origin_w) / 2), (int)((resize_h - origin_h) / 2), origin_w, origin_h);
    zoomed_img = resized_img(roi);
  } else {          // zoom out
    cv::Mat roi(zoomed_img, cv::Rect((int)((origin_w - resize_w) / 2), (int)((origin_h - resize_h) / 2), resize_w, resize_h));
    resized_img.copyTo(roi);
    zoomed_img.copyTo(cv_img);
  }

  zoomed_img.copyTo(cv_img);

  if (m_show_info) {
    LOG(INFO) << "* Pixel for Zoom : " << pixel;
  }
}

  INSTANTIATE_CLASS(DataAugmenter);
} //namespace caffe
