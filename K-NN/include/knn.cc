#include "knn.hpp"
#include <algorithm> // For std::sort
#include <cmath>     // For std::sqrt
#include <unordered_map> // For voting mechanism

// Constructor with k parameter
KNN::KNN(int k_val) : k(k_val) {}

// Default Constructor
KNN::KNN() : k(3) {} // Default k=3

// Destructor
KNN::~KNN() {}

// Setter for k
void KNN::set_k(int val) {
    k = val;
}

// Setter for training data
void KNN::set_training_data(const std::vector<std::unique_ptr<data>>& vect) {
    trainingData.clear();
    trainingData.reserve(vect.size());
    for(const auto& data_ptr : vect) {
        trainingData.push_back(data_ptr.get());
    }
}

// Setter for test data
void KNN::set_test_data(const std::vector<std::unique_ptr<data>>& vect) {
    testDataSet.clear();
    testDataSet.reserve(vect.size());
    for(const auto& data_ptr : vect) {
        testDataSet.push_back(data_ptr.get());
    }
}

// Setter for validation data
void KNN::set_validation_data(const std::vector<std::unique_ptr<data>>& vect) {
    validationDataSet.clear();
    validationDataSet.reserve(vect.size());
    for(const auto& data_ptr : vect) {
        validationDataSet.push_back(data_ptr.get());
    }
}

// Calculate Euclidean distance between two data points
double KNN::calculate_distance(const data* query_point, const data* input) {
    double sum = 0.0;
    const auto& query_features = query_point->get_normalized_feature_vector();
    const auto& input_features = input->get_normalized_feature_vector();
    size_t size = query_features.size();
    for(size_t i = 0; i < size; ++i) {
        double diff = static_cast<double>(query_features[i]) - static_cast<double>(input_features[i]);
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Find k nearest neighbors for a given query point
void KNN::find_k_nearest_neighbors(const data* query_point) {
    neighbors.clear();
    // Vector to hold pairs of (distance, data*)
    std::vector<std::pair<double, data*>> distance_vector;
    distance_vector.reserve(trainingData.size());

    // Calculate distance from query_point to each training data point
    for(auto train_ptr : trainingData) {
        double dist = calculate_distance(query_point, train_ptr);
        distance_vector.emplace_back(dist, train_ptr);
    }

    // Sort the distance_vector based on distance
    std::sort(distance_vector.begin(), distance_vector.end(),
              [](const std::pair<double, data*>& a, const std::pair<double, data*>& b) {
                  return a.first < b.first;
              });

    // Select the top k nearest neighbors
    for(int i = 0; i < k && i < static_cast<int>(distance_vector.size()); ++i) {
        neighbors.push_back(distance_vector[i].second);
    }
}

// Predict the label for a given query point
int KNN::predict(const data* query_point) {
    find_k_nearest_neighbors(query_point);

    // Voting mechanism using unordered_map
    std::unordered_map<int, int> vote_map;
    for(auto neighbor_ptr : neighbors) {
        vote_map[neighbor_ptr->get_enumerated_label()]++;
    }

    // Find the label with the maximum votes
    int predicted_label = -1;
    int max_votes = -1;
    for(const auto& pair : vote_map) {
        if(pair.second > max_votes) {
            max_votes = pair.second;
            predicted_label = pair.first;
        }
    }

    return predicted_label;
}

// Validate the KNN model using the validation dataset
double KNN::validate() {
    if(validationDataSet.empty()) {
        std::cerr << "Validation dataset is empty." << std::endl;
        return 0.0;
    }

    int correct = 0;
    for(const auto& query_ptr : validationDataSet) {
        int prediction = predict(query_ptr);
        if(prediction == query_ptr->get_enumerated_label()) {
            correct++;
        }
    }

    double accuracy = static_cast<double>(correct) / validationDataSet.size();
    std::cout << "Validation Accuracy: " << accuracy * 100.0 << "%" << std::endl;
    return accuracy;
}

// Test the KNN model using the test dataset
double KNN::test() {
    if(testDataSet.empty()) {
        std::cerr << "Test dataset is empty." << std::endl;
        return 0.0;
    }

    int correct = 0;
    for(const auto& query_ptr : testDataSet) {
        int prediction = predict(query_ptr);
        if(prediction == query_ptr->get_enumerated_label()) {
            correct++;
        }
    }

    double accuracy = static_cast<double>(correct) / testDataSet.size();
    std::cout << "Test Accuracy: " << accuracy * 100.0 << "%" << std::endl;
    return accuracy;
}

// Optional: Getter for neighbors
const std::vector<data*>& KNN::get_neighbors() const {
    return neighbors;
}
