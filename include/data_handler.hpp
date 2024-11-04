#ifndef __DATA_HANDLER_HPP
#define __DATA_HANDLER_HPP

#include "data.hpp"    // Include the full definition of 'data'
#include <fstream>
#include <string>
#include <map>
#include <future>      // For threading
#include <cstdint>     // For uint8_t


class data_handler
{
    // Using unique_ptr to manage the vectors
    std::unique_ptr<std::vector<std::unique_ptr<data>>> data_array;          // Vector to store all data pre-split
    std::unique_ptr<std::vector<std::unique_ptr<data>>> training_data;       // Vector to store training data for training the model
    std::unique_ptr<std::vector<std::unique_ptr<data>>> test_data;           // Vector to store testing data for final evaluation
    std::unique_ptr<std::vector<std::unique_ptr<data>>> validation_data;     // Vector to store validation data for intermediate evaluation

    int class_counts;
    int feature_vector_size;

    std::map<uint8_t, int> classFromInt;
    std::map<std::string, int> classFromString; // String key

    // Temporary storage for images and labels
    std::vector<std::vector<uint8_t>> temp_image_data;
    std::vector<uint8_t> temp_label_data;

public:
    const double TRAIN_SET_PERCENT = 0.75;
    const double TEST_SET_PERCENT = 0.20;
    const double VALIDATION_SET_PERCENT = 0.05;

    // Constructor
    data_handler();
    // Destructor
    ~data_handler();

    // Load data from file
    void read_feature_vector(const std::string& path);
    // Read labels from file
    void read_feature_labels(const std::string& path);
    // Combine images and labels into data_array
    void combine_data();
    // Split data into training, test, and validation sets according to the percentages
    void split_data();
    void count_classes();
    void normalize();
    void print();

    int get_class_counts();
    int get_data_array_size();
    int get_training_data_size();
    int get_test_data_size();
    int get_validation_size();

    // Helper function to read big-endian uint32_t
    uint32_t read_uint32(std::ifstream& file);

    // Getters (Accessors)
    // Returning raw pointers to the vectors for compatibility
    // Alternatively, you can return references or smart pointers if possible
    std::vector<std::unique_ptr<data>>* get_training_data();
    std::vector<std::unique_ptr<data>>* get_test_data();
    std::vector<std::unique_ptr<data>>* get_validation_data();
};

#endif

