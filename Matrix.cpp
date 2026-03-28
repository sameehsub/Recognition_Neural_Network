#include "Matrix.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>

// constructor
Matrix::Matrix(int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("Matrix dimensions must be positive integers.");
    }
    _rows = rows;
    _cols = cols;
    _data = new float[rows * cols]();
}

// default constructor
Matrix::Matrix() : _rows(1), _cols(1), _data(new float[1]()) {}

// copy constructor
Matrix::Matrix(const Matrix& other) : _rows(other._rows), _cols(other._cols), _data(new float[other._rows * other._cols])
{
    for (int i = 0; i < _rows * _cols; i++)
    {
        _data[i] = other._data[i];
    }
}

// destructor
Matrix::~Matrix()
{
    delete[] _data;
}

// assignment operator
Matrix& Matrix::operator=(const Matrix& other)
{
    if (this == &other) return *this;
    delete[] _data;
    _rows = other._rows;
    _cols = other._cols;
    _data = new float[_rows * _cols];
    for (int i = 0; i < _rows * _cols; i++) {
        _data[i] = other._data[i];
    }
    return *this;
}

// matrix addition (+=)
Matrix& Matrix::operator+=(const Matrix& other) {
    if (_rows != other._rows || _cols != other._cols) {
        throw std::runtime_error("Matrices must have the same dimensions for addition.");
    }
    for (int i = 0; i < _rows * _cols; i++) {
        _data[i] += other._data[i];
    }
    return *this;
}

// matrix addition (+)
Matrix Matrix::operator+(const Matrix& other) const
{
    if (_rows != other._rows || _cols != other._cols)
    {
        throw std::runtime_error("Matrices must have the same dimensions for addition.");
    }

    Matrix result(_rows, _cols);

    for (int i = 0; i < _rows * _cols; i++)
    {
        result._data[i] = _data[i] + other._data[i];
    }
    return result;
}

// matrix multiplication
Matrix Matrix::operator*(const Matrix& other) const
{
    if (_cols != other._rows)
    {
        throw std::runtime_error("Number of columns in the first matrix must equal the number of rows in the second. ");
    }

    Matrix result(_rows, other._cols);
    for (int i=0; i<_rows; i++){
        for (int j=0; j<other._cols; j++){
            float sum = 0.0f;
            for (int k=0; k <_cols; k++){
                sum += (*this)(i,k) * other(k,j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

// scalar multiplication from the right
Matrix Matrix::operator*(float scalar) const
{
    if (_rows<=0 || _cols<= 0)
    {
        throw std::runtime_error("Cannot scale the matrix.");
    }
    Matrix result(_rows, _cols);
    for (int i = 0; i < _rows * _cols; i++) {
        result._data[i] = _data[i] * scalar;
    }
    return result;
}

// scalar multiplication from the left
Matrix operator*(float scalar, const Matrix& self) {
    return self * scalar;
}

float Matrix::operator()(int row, int col) const {
    if (row < 0 || row >= _rows || col < 0 || col >= _cols) {
        throw std::runtime_error("Matrix indices out of range.");
    }
    return _data[row * _cols + col];
}

float& Matrix::operator()(int row, int col) {
    if (row < 0 || row >= _rows || col < 0 || col >= _cols) {
        throw std::runtime_error("Matrix indices out of range.");
    }
    return _data[row * _cols + col];
}

float Matrix::operator[](int n) const {
    if (n < 0 || n >= _rows * _cols) {
        throw std::runtime_error("Matrix index out of range.");
    }
    return _data[n];
}

float& Matrix::operator[](int n) {
    if (n < 0 || n >= _rows * _cols) {
        throw std::runtime_error("Matrix index out of range.");
    }
    return _data[n];
}

// Output stream operator for printing
std::ostream& operator<<(std::ostream& os, const Matrix& m) {
    if (m._rows <= 0 || m._cols <= 0) {
        throw std::runtime_error("Cannot print an empty matrix.");
    }
    for (int i = 0; i < m._rows; ++i) {
        for (int j = 0; j < m._cols; ++j) {
            os << m(i, j) << " ";
        }
        os << std::endl;
    }
    return os;
}


// Input stream operator for reading binary data
std::istream& operator>>(std::istream& is, Matrix& m) {
    for (int i = 0; i < m._rows * m._cols; ++i) {
        if (!is.read(reinterpret_cast<char*>(&m._data[i]), sizeof(float))) {
            throw std::runtime_error("Failed to read matrix data from input stream.");
        }
    }
    return is;
}

int Matrix::get_rows() const {
    return _rows;
}

int Matrix::get_cols() const {
    return _cols;
}

void Matrix::plain_print() const {
    if (_rows == 0 || _cols == 0) {
        throw std::runtime_error("Cannot print an empty matrix.");
    }
    for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j < _cols; ++j) {
            std::cout << _data[i * _cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

Matrix& Matrix::transpose() {
    if (_rows == 0 || _cols == 0) {
        throw std::runtime_error("Cannot transpose an empty matrix.");
    }
    float* newData = new float[_rows * _cols];
    for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j < _cols; ++j) {
            newData[j * _rows + i] = _data[i * _cols + j];
        }
    }
    delete[] _data;
    _data = newData;
    int temp = _rows;
    _rows = _cols;
    _cols = temp;
    return *this;
}

Matrix& Matrix::vectorize() {
    if (_rows == 0 || _cols == 0) {
        throw std::runtime_error("Cannot vectorize an empty matrix.");
    }
    _rows *= _cols;
    _cols = 1;
    return *this;
}


Matrix Matrix::dot(const Matrix& m) const {
    if (_rows != m._rows || _cols != m._cols) {
        throw std::runtime_error("Matrices must have the same dimensions for dot product.");
    }
    Matrix ret(_rows, _cols);
    for (int i = 0; i < _rows; i++) {
        for (int j = 0; j < _cols; j++) {
            ret(i, j) = (*this)(i, j) * m(i, j);
        }
    }
    return ret;
}

float Matrix::norm() const {
    if (_rows == 0 || _cols == 0) {
        throw std::runtime_error("Cannot compute the norm of an empty matrix.");
    }
    float sum = 0.0f;
    for (int i = 0; i < _rows * _cols; ++i) {
        sum += _data[i] * _data[i];
    }
    return sqrt(sum);
}

float Matrix::sum() const {
    float total = 0.0f;
    for (int i = 0; i < _rows * _cols; ++i) {
        total += _data[i];
    }
    return total;
}

int Matrix::argmax() const {
    if (_rows * _cols == 0) {
        throw std::runtime_error("Cannot find argmax of an empty matrix.");
    }
    int maxIndex = 0;
    float maxValue = _data[0];
    for (int i = 1; i < _rows * _cols; ++i) {
        if (_data[i] > maxValue) {
            maxValue = _data[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}

Matrix Matrix::rref() const {
    Matrix result(*this);

    int lead = 0;
    for (int r = 0; r < _rows; ++r) {
        if (lead >= _cols) {
            break;
        }

        int i = r;
        while (result(i, lead) == 0) {
            i++;
            if (i == _rows) {
                i = r;
                lead++;
                if (lead == _cols) {
                    return result;
                }
            }
        }

        for (int k = 0; k < _cols; ++k) {
            std::swap(result(i, k), result(r, k));
        }

        float leadVal = result(r, lead);
        if (leadVal != 0) {
            for (int k = 0; k < _cols; ++k) {
                result(r, k) /= leadVal;
            }
        }

        for (int i = 0; i < _rows; ++i) {
            if (i != r) {
                float factor = result(i, lead);
                for (int k = 0; k < _cols; ++k) {
                    result(i, k) -= factor * result(r, k);
                }
            }
        }
        lead++;
    }

    return result;
}














