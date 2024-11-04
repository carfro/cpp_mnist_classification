#include "data_handler.hpp"
#include "data.hpp" 
#include <algorithm>
#include <random>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <omp.h> // For OpenMP
#include <iomanip> // For std::fixed and std::setprecision


// Constructor
data_handler::data_handler()
    : data_array(std::make_unique<std::vector<std::unique_ptr<data>>>()),
      training_data(std::make_unique<std::vector<std::unique_ptr<data>>>()),
      test_data(std::make_unique<std::vector<std::unique_ptr<data>>>()),
      validation_data(std::make_unique<std::vector<std::unique_ptr<data>>>())
{
    // Vectors are initialized as empty; no further action needed
}

// Destructor
data_handler::~data_handler()
{
    // No manual deletion needed as smart pointers handle memory management
}

// Helper function to read big-endian uint32_t
uint32_t data_handler::read_uint32(std::ifstream& file)
{
    uint32_t result = 0;
    unsigned char bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);
    if (!file)
    {
        std::cerr << "Error reading uint32 from file." << std::endl;
        exit(1);
    }
    result = (static_cast<uint32_t>(bytes[0]) << 24) |
             (static_cast<uint32_t>(bytes[1]) << 16) |
             (static_cast<uint32_t>(bytes[2]) << 8) |
             (static_cast<uint32_t>(bytes[3]));
    return result;
}

// Load data from file
void data_handler::read_feature_vector(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        std::cerr << "Could not open image file: " << path << std::endl;
        exit(1);
    }

    uint32_t magic_number = read_uint32(file);
    uint32_t num_images = read_uint32(file);
    uint32_t num_rows = read_uint32(file);
    uint32_t num_cols = read_uint32(file);

    size_t image_size = num_rows * num_cols;

    // Read all images into a buffer
    std::vector<uint8_t> images(num_images * image_size);
    file.read(reinterpret_cast<char*>(images.data()), images.size());

    if (!file)
    {
        std::cerr << "Error reading image data from file." << std::endl;
        exit(1);
    }

    // Resize temp_image_data
    temp_image_data.resize(num_images);

    // Fill temp_image_data
    for (size_t i = 0; i < num_images; ++i)
    {
        temp_image_data[i].assign(images.begin() + i * image_size, images.begin() + (i + 1) * image_size);
    }

    std::cout << "Successfully read and stored " << temp_image_data.size() << " feature vectors." << std::endl;
}

// Read labels from file
void data_handler::read_feature_labels(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        std::cerr << "Could not open label file: " << path << std::endl;
        exit(1);
    }

    uint32_t magic_number = read_uint32(file);
    uint32_t num_labels = read_uint32(file);

    // Read all labels into temp_label_data
    temp_label_data.resize(num_labels);
    file.read(reinterpret_cast<char*>(temp_label_data.data()), num_labels);

    if (!file)
    {
        std::cerr << "Error reading label data from file." << std::endl;
        exit(1);
    }

    std::cout << "Successfully read and stored labels." << std::endl;
}

// Combine images and labels into data_array
void data_handler::combine_data()
{
    if (temp_image_data.size() != temp_label_data.size())
    {
        std::cerr << "Mismatch between number of images and labels." << std::endl;
        exit(1);
    }

    data_array->reserve(temp_image_data.size());

    for (size_t i = 0; i < temp_image_data.size(); ++i)
    {
        auto d = std::make_unique<data>();
        d->set_feature_vector(temp_image_data[i]);
        d->set_label(temp_label_data[i]);
        data_array->emplace_back(std::move(d));
    }

    // Clear temporary data
    temp_image_data.clear();
    temp_label_data.clear();
}

// Split data into training, test, and validation sets according to the percentages
void data_handler::split_data()
{
    size_t total_size = data_array->size();
    size_t train_size = static_cast<size_t>(total_size * TRAIN_SET_PERCENT);
    size_t test_size = static_cast<size_t>(total_size * TEST_SET_PERCENT);
    size_t valid_size = total_size - train_size - test_size;

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 g(rd());

    // Shuffle the data_array
    std::shuffle(data_array->begin(), data_array->end(), g);

    // Split the data by moving unique_ptrs to respective vectors
    // Training Data
    for (size_t i = 0; i < train_size; ++i)
    {
        training_data->emplace_back(std::move((*data_array)[i]));
    }

    // Test Data
    for (size_t i = train_size; i < train_size + test_size; ++i)
    {
        test_data->emplace_back(std::move((*data_array)[i]));
    }

    // Validation Data
    for (size_t i = train_size + test_size; i < total_size; ++i)
    {
        validation_data->emplace_back(std::move((*data_array)[i]));
    }

    // Clear the original data_array as ownership has been moved
    data_array->clear();

    std::cout << "Training Data Size: " << training_data->size() << "." << std::endl;
    std::cout << "Test Data Size: " << test_data->size() << "." << std::endl;
    std::cout << "Validation Data Size: " << validation_data->size() << "." << std::endl;
}

void data_handler::count_classes()
{
    // Loop over each data point and build the class mapping
    for (const auto& data_ptr : *data_array)
    {
        uint8_t label = data_ptr->get_label();
        auto result = classFromInt.try_emplace(label, classFromInt.size());
        data_ptr->set_enumerated_label(result.first->second);
    }

    // Set the total number of unique classes
    class_counts = static_cast<int>(classFromInt.size());

    // Update each data point with the class vector
    for (const auto& data_ptr : *data_array)
    {
        data_ptr->set_class_vector(class_counts);
    }

    std::cout << "Successfully Extracted " << class_counts << " Unique Classes." << std::endl;
}


void data_handler::normalize()
{
    size_t n = data_array->size();
    if(n == 0) {
        std::cerr << "No data to normalize." << std::endl;
        return;
    }

    // Initialize mean and M2 vectors using float for efficiency
    std::vector<float> mean(feature_vector_size, 0.0f);
    std::vector<float> M2(feature_vector_size, 0.0f);

    // Compute mean and M2 using Welford's algorithm in a single pass
    // Parallelize the outer loop
    #pragma omp parallel for
    for(long long idx = 0; idx < static_cast<long long>(n); ++idx)
    {
        const auto& data_ptr = (*data_array)[idx];
        const auto& feature_vector = data_ptr->get_feature_vector();
        for(size_t i = 0; i < feature_vector_size; ++i)
        {
            float x = static_cast<float>(feature_vector[i]);
            float delta = x - mean[i];
            #pragma omp atomic
            mean[i] += delta / static_cast<float>(n);
            float delta2 = x - mean[i];
            #pragma omp atomic
            M2[i] += delta * delta2;
        }
    }

    // Compute standard deviation
    std::vector<float> std_dev(feature_vector_size, 0.0f);
    #pragma omp parallel for
    for(size_t i = 0; i < feature_vector_size; ++i)
    {
        if(n < 2)
            std_dev[i] = 0.0f;
        else
            std_dev[i] = std::sqrt(M2[i] / static_cast<float>(n - 1));
    }

    // Handle zero standard deviation to avoid division by zero
    #pragma omp parallel for
    for(size_t i = 0; i < feature_vector_size; ++i)
    {
        if(std_dev[i] == 0.0f)
        {
            std_dev[i] = 1.0f; // Alternatively, assign a small epsilon value
            #pragma omp critical
            {
                std::cerr << "Feature " << i << " has zero standard deviation. Adjusted to 1.0 to avoid division by zero." << std::endl;
            }
        }
    }

    // Normalize the feature vectors
    #pragma omp parallel for
    for(long long idx = 0; idx < static_cast<long long>(n); ++idx)
    {
        auto& data_ptr = (*data_array)[idx];
        auto normalized_feature_vector = std::make_unique<std::vector<float>>(feature_vector_size, 0.0f);
        const auto& feature_vector = data_ptr->get_feature_vector();
        for(size_t j = 0; j < feature_vector_size; ++j)
        {
            normalized_feature_vector->at(j) = (static_cast<float>(feature_vector[j]) - mean[j]) / std_dev[j];
        }
        data_ptr->set_normalized_feature_vector(std::move(normalized_feature_vector));
    }

    std::cout << "Data normalization completed successfully." << std::endl;
}


void data_handler::print()
{
    // Lambda function to print a dataset
    auto print_dataset = [](const std::string& dataset_name, const std::vector<std::unique_ptr<data>>& dataset) {
        std::cout << dataset_name << " Data:\n";
        std::cout << std::fixed << std::setprecision(3); // Set floating-point precision
        for(const auto& data_ptr : dataset)
        {
            const auto& normalized_features = data_ptr->get_normalized_feature_vector();
            for(const auto& value : normalized_features)
            {
                std::cout << value << ",";
            }
            std::cout << " -> " << static_cast<int>(data_ptr->get_label()) << "\n";
        }
        std::cout << std::defaultfloat << "\n"; // Reset to default formatting and add a newline for readability
    };

    // Print each dataset using the lambda
    print_dataset("Training", *training_data);
    print_dataset("Test", *test_data);
    print_dataset("Validation", *validation_data);
}

// Getters (Accessors)
std::vector<std::unique_ptr<data>>* data_handler::get_training_data()
{
    return training_data.get();
}

std::vector<std::unique_ptr<data>>* data_handler::get_test_data()
{
    return test_data.get();
}

std::vector<std::unique_ptr<data>>* data_handler::get_validation_data()
{
    return validation_data.get();
}

int data_handler::get_class_counts()
{
    return class_counts;
}

int data_handler::get_data_array_size()
{
    return static_cast<int>(data_array->size());
}

int data_handler::get_training_data_size()
{
    return static_cast<int>(training_data->size());
}

int data_handler::get_test_data_size()
{
    return static_cast<int>(test_data->size());
}

int data_handler::get_validation_size()
{
    return static_cast<int>(validation_data->size());
}

// Updated main function using smart pointers
int main()
{
    // Use unique_ptr to manage data_handler
    auto dh = std::make_unique<data_handler>();

    // Read images and labels in parallel
    std::future<void> future_images = std::async(std::launch::async, &data_handler::read_feature_vector, dh.get(), "./data/train-images.idx3-ubyte");
    std::future<void> future_labels = std::async(std::launch::async, &data_handler::read_feature_labels, dh.get(), "./data/train-labels.idx1-ubyte");

    // Wait for both to finish
    future_images.get();
    future_labels.get();

    // Combine data
    dh->combine_data();

    dh->count_classes();

    dh->split_data();

    // No need to manually delete dh; it will be automatically cleaned up

    return 0;
}
