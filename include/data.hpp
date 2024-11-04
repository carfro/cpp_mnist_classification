#ifndef __DATA_HPP
#define __DATA_HPP

#include <vector>
#include <memory> // for std::unique_ptr
#include <iostream>
#include "stdint.h"
#include "stdio.h"

class data
{

  std::unique_ptr<std::vector<uint8_t>> feature_vector; // Smart pointer to the feature vector
  std::unique_ptr<std::vector<float>> normalized_feature_vector; // Smart pointer to the normalized feature vector
  std::unique_ptr<std::vector<int>> class_vector;       // Smart pointer to the class vector
  uint8_t label;                                        // Label of the data, actual class
  int enum_label;                                       // Label of the data, enumerated class
  double distance;                                      // Distance from query point

public:
  data();
  ~data();

  // Setters (Mutators)
  void set_feature_vector(const std::vector<uint8_t> &);
  void set_normalized_feature_vector(std::unique_ptr<std::vector<float>> vect);


  void append_to_feature_vector(const uint8_t *elements, size_t size);
  void append_to_feature_vector(const double *elements, size_t size);

  void print_vector();
  void print_normalized_vector();


  void set_label(uint8_t);
  void set_enumerated_label(int);
  void set_class_vector(int); // Declare the function

  // Getters (Accessors)
  int get_feature_vector_size();
  uint8_t get_label();
  uint8_t get_enumerated_label();
  double get_distance() const;

  const std::vector<uint8_t>& get_feature_vector() const;
  const std::vector<int>& get_class_vector() const;
  const std::vector<float>& get_normalized_feature_vector() const;
};

#endif
