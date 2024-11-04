#ifndef __KNN_HPP
#define __KNN_HPP

#include <vector>
#include <memory>
#include "../../include/data_handler.hpp" // Adjust the path as per your project structure

class KNN 
{
private:
    // Number of neighbors to consider
    int k;

    // Data sets as raw pointers (non-owning references)
    std::vector<data*> neighbors;
    std::vector<data*> trainingData;
    std::vector<data*> testDataSet;
    std::vector<data*> validationDataSet;

public:
    // Constructors and Destructor
    KNN(int k_val);
    KNN();
    ~KNN();

    // Core KNN functionality
    void find_k_nearest_neighbors(const data* query_point);
    int predict(const data* query_point);
    double calculate_distance(const data* query_point, const data* input);

    // Data setters
    void set_training_data(const std::vector<std::unique_ptr<data>>& vect);
    void set_test_data(const std::vector<std::unique_ptr<data>>& vect);
    void set_validation_data(const std::vector<std::unique_ptr<data>>& vect);
    void set_k(int val);

    // Evaluation
    double validate();
    double test();

    // Optional: Getter for neighbors
    const std::vector<data*>& get_neighbors() const;
};

#endif // __KNN_HPP
