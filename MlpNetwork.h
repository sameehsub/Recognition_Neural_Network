// MlpNetwork.h
#ifndef MLPNETWORK_H
#define MLPNETWORK_H

//******** Don't touch the next lines! ********
#include "Matrix.h"
#include "Dense.h"
#include "Activation.h"

#define MLP_SIZE 4


const matrix_dims img_dims = {28, 28};
const matrix_dims weights_dims[] = {{128, 784},
									{64,  128},
									{20,  64},
									{10,  20}};
const matrix_dims bias_dims[] = {{128, 1},
								 {64,  1},
								 {20,  1},
								 {10,  1}};
/**
 * @struct digit
 * @brief Identified (by Mlp network) digit with
 *        the associated probability.
 * @var value - Identified digit value
 * @var probability - identification probability
 */
typedef struct digit {
	unsigned int value;
	float probability;
} digit;
//******** From now-on you can write your code ********

class MlpNetwork {
private:
	Dense _layers[MLP_SIZE]; // Array to store the 4 layers of the network

public:
	/**
	 * Constructor
	 * @param weights Array of 4 weight matrices
	 * @param biases Array of 4 bias matrices
	 */
	MlpNetwork();
	MlpNetwork(Matrix weights[], Matrix biases[], ActFunc acts[]);


	/**
	 * Applies the entire network on the input matrix
	 * @param input Input matrix to the network
	 * @return The output digit struct with the most probable digit and its probability
	 */
	digit operator()(const Matrix& input) const;
};


#endif // MLPNETWORK_H