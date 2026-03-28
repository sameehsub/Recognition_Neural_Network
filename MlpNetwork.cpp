#include "MlpNetwork.h"

MlpNetwork::MlpNetwork(Matrix weights[], Matrix biases[], ActFunc acts[])
{
    for (int i = 0; i < MLP_SIZE; ++i)
    {
        _layers[i] = Dense(weights[i], biases[i], acts[i]);
    }
}


digit MlpNetwork::operator()(const Matrix& input) const {
    Matrix result = input;
    for (int i = 0; i < MLP_SIZE; ++i) {
        result = _layers[i](result);
    }

    float maxProb = result[0];
    unsigned int maxIndex = 0;

    for (int i = 1; i < result.get_rows(); ++i) {
        if (result[i] > maxProb) {
            maxProb = result[i];
            maxIndex = i;
        }
    }

    return {maxIndex, maxProb};
}

MlpNetwork::MlpNetwork()
{
    Matrix w(1, 1);
    Matrix b(1, 1);

    w(0, 0) = 0.0f;
    b(0, 0) = 0.0f;

    for (int i = 0; i < MLP_SIZE; ++i)
    {
        _layers[i] = Dense(w, b, activation::relu);
    }
}


