#include "Dense.h"


Dense::Dense() : _weights(Matrix(1, 1)), _bias(Matrix(1, 1)), _func(nullptr) {}


Dense::Dense(Matrix &weights, Matrix &bias, ActFunc func)
    : _weights(weights), _bias(bias), _func(func) {}

Matrix Dense::get_weights() const {
    return _weights;
}

Matrix Dense::get_bias() const {
    return _bias;
}

ActFunc Dense::get_activation() const {
    return _func;
}

// Apply the Dense layer to an input matrix

Matrix Dense::operator()(const Matrix &input) const {
    // Ensure input dimensions are valid for matrix multiplication
    if (input.get_rows() != _weights.get_cols()) {
        throw "Input dimensions do not match the weight dimensions for multiplication.";
    }

    // Perform weighted sum: weights * input + bias
    Matrix weighted_sum = (_weights * input) + _bias;

    // Apply the activation function
    return _func(weighted_sum);
}