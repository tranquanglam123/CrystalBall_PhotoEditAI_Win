#ifndef _FILTERING_LOW_PASS_FILTER_H_
#define _FILTERING_LOW_PASS_FILTER_H_

#include <memory>

class LowPassFilter {
 public:
  explicit LowPassFilter(float alpha);

  float Apply(float value);

  float ApplyWithAlpha(float value, float alpha);

  bool HasLastRawValue();

  float LastRawValue();

  float LastValue();

 private:
  void SetAlpha(float alpha);

  float raw_value_;
  float alpha_;
  float stored_value_;
  bool initialized_;
};
#endif