#ifndef DENSE_H
#define DENSE_H

#include "Matrix.h"
#include "Activation.h"

class Dense{
private:
    Matrix _weights;
    Matrix _bias;
    ActFunc _func;

public:
    // Default constructor
    Dense();

    // Parameterized constructor
    Dense(Matrix &weights, Matrix &bias, ActFunc func);

    // Methods
    Matrix get_weights() const;
    Matrix get_bias() const;
    ActFunc get_activation() const;

    // Operator
    Matrix operator()(const Matrix &input) const;
};

#endif // DENSE_H