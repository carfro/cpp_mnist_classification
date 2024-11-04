#include "data.hpp"

data::data()
    : feature_vector(std::make_unique<std::vector<uint8_t>>()),
      class_vector(nullptr)
{
}

data::~data() = default; // Let unique_ptr automatically manage memory

void data::set_feature_vector(const std::vector<uint8_t>& vect)
{
    feature_vector = std::make_unique<std::vector<uint8_t>>(vect);
}

void data::set_normalized_feature_vector(std::unique_ptr<std::vector<float>> vect)
{
    normalized_feature_vector = std::move(vect);
}

const std::vector<float>& data::get_normalized_feature_vector() const
{
    return *normalized_feature_vector;
}

void data::append_to_feature_vector(const double* elements, size_t size)
{
    normalized_feature_vector->insert(normalized_feature_vector->end(), elements, elements + size);
}

void data::append_to_feature_vector(const uint8_t* elements, size_t size)
{
    feature_vector->insert(feature_vector->end(), elements, elements + size);
}

void data::set_label(uint8_t lbl)
{
    label = lbl;
}

void data::set_enumerated_label(int lbl)
{
    enum_label = lbl;
}

void data::set_class_vector(int class_counts)
{
    class_vector = std::make_unique<std::vector<int>>(class_counts, 0);
    class_vector->at(enum_label) = 1;
}

// Getters (Accessors)
int data::get_feature_vector_size()
{
    return feature_vector->size();
}

uint8_t data::get_label()
{
    return label;
}

uint8_t data::get_enumerated_label()
{
    return enum_label;
}

const std::vector<uint8_t>& data::get_feature_vector() const
{
    return *feature_vector;
}

const std::vector<int>& data::get_class_vector() const
{
    return *class_vector;
}

void data::print_vector()
{
    std::cout << "[ ";
    for (const auto& elem : *feature_vector)
    {
        std::cout << static_cast<int>(elem) << " ";
    }
    std::cout << "]" << std::endl;
}

void data::print_normalized_vector()
{
    std::cout << "[ ";
    for (const auto& elem : *normalized_feature_vector)
    {
        std::cout << elem << " ";
    }
    std::cout << "]" << std::endl;
}

double data::get_distance() const
{
    return distance;
}
