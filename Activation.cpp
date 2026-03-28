#include "Activation.h"
#include <cmath>
#include <limits>

namespace activation
{
    Matrix relu(const Matrix& m)
    {
        Matrix result(m.get_rows(), m.get_cols());
        for (int i=0; i<m.get_rows(); i++)
        {
            for (int j=0; j<m.get_cols(); j++)
            {
                result(i,j) = std::max(0.0f, m(i,j));
            }
        }
        return result;
    }

    Matrix softmax(const Matrix& m) {
        Matrix result(m.get_rows(), m.get_cols());

        float max_val = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < m.get_rows(); ++i)
            for (int j = 0; j < m.get_cols(); ++j)
                max_val = std::max(max_val, m(i, j));

        float sum = 0.0f;
        for (int i = 0; i < m.get_rows(); ++i)
            for (int j = 0; j < m.get_cols(); ++j)
            {
                result(i, j) = std::exp(m(i, j) - max_val);
                sum += result(i, j);
            }

        for (int i = 0; i < m.get_rows(); ++i)
            for (int j = 0; j < m.get_cols(); ++j)
                result(i, j) /= sum;

        return result;
    }
}