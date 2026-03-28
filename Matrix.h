// Matrix.h
#ifndef MATRIX_H
#define MATRIX_H
#include <iostream>


//******** Don't touch the next lines! ********
//you may use the struct for your convenience, you don't really have to.
typedef struct matrix_dims {
    int rows, cols;
}matrix_dims;
//******** From now on, you can write your code ********

// Insert Matrix class here...
class Matrix
{
private:
    int _rows;
    int _cols;
    float* _data;

public:

    // ---------- CONSTRUCTORS ---------- //

    // constructor
    Matrix(int rows, int cols);

    // default constructor
    Matrix();

    // copy constructor
    Matrix(const Matrix& other);

    //destructor
    ~Matrix();


    // ---------- OPERATORS ---------- //

    // assignment operator
    Matrix& operator=(const Matrix& other);

    // matrix addition (+=)
    Matrix& operator+=(const Matrix& other);

    // matrix addition (+)
    Matrix operator+(const Matrix& other) const;

    // matrix multiplication
    Matrix operator*(const Matrix& other) const;

    // scalar multiplication from the right
    Matrix operator*(float scalar) const;

    // scalar multiplication from the left
    friend Matrix operator*(float scalar, const Matrix& self);

    // parenthesis indexing
    float operator()(int row, int col) const;
    float& operator()(int row, int col);

    // brackets
    float operator[](int n) const;
    float& operator[](int n);

    // Output stream operator for printing
    friend std::ostream& operator<<(std::ostream& os, const Matrix& m);

    // Input stream operator for reading binary data
    friend std::istream& operator>>(std::istream& is, Matrix& m);



    // ---------- GENERAL METHODS ---------- //
    int get_rows() const;
    int get_cols() const;
    void plain_print() const;

    // ---------- METH METHODS ---------- //

    Matrix& transpose();
    Matrix& vectorize();
    Matrix dot(const Matrix& other) const;
    float norm() const;
    float sum() const;
    int argmax() const;
    Matrix rref() const;

};

#endif //MATRIX_H