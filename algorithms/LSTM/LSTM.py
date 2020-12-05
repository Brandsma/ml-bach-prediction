# Difference LSTM to ESN --> LSTM is desgined to deal with the vanishing gradient problem, encountered when training RNN's. 
# Relative insensitivity to gap length is an advantage of LSTM over RNNs, hidden Markov models and other sequence learning methods in numerous applications.
import numpy as np
from math import e

class LSTM():

    def __init__(self, h_size, d_size):
        # model values
        self.t = 0

        # cell state content
        self.C_t = np.zeros(h_size)
        
        # input from previous cell / output to user and next cell
        self.h_t = np.zeros(h_size)

        # weight matrices | f : forget, i : input, o : output, c : cell
        self.W_f = np.zeros((h_size, d_size))
        self.U_f = np.zeros((h_size, h_size))
        self.W_i = np.zeros((h_size, d_size))
        self.U_i = np.zeros((h_size, h_size))
        self.W_o = np.zeros((h_size, d_size))
        self.U_o = np.zeros((h_size, h_size))
        self.W_c = np.zeros((h_size, d_size))
        self.U_c = np.zeros((h_size, h_size))

        # biases
        self.b_f = np.zeros(h_size)
        self.b_i = np.zeros(h_size)
        self.b_o = np.zeros(h_size)
        self.b_c = np.zeros(h_size)

    # Regulators -- aka gates -- of the flow of information inside the LSTM unit.
    def matrix_calculation(self, h, x, W, U, b):
        """Core matrix calculation operation used throughout the model
        input : 
            h = h_t-1 (output from previous round), vector of size d
            x = x_t   (input),                      vector of size h
            W = Weight matrix W,                    size h * d
            U = Weight matrix U                     size h * h
            b = Bias vector                         size h
        output:
            (W * x) + (U * h) + b 
            output is raw, will be processed through sigmodal or tanh transformation
        """
        # product of matrix h * d and vector d -> vector h
        weighted_output = W * h
        # product of matrix h * h and vector h -> vector h
        weighted_input = U * x
        return (weighted_input + weighted_output + b) # Vector of size h

    # The input gate controls the extent to which a new value flows into the cell
    def input_gate(self, h_t_0, x_t, W_i, U_i, b_i, W_c, U_c, b_c):
        """Decide what information to put in the cell
            The sigmoid layer decides which values to update
            The tanh layer creates a vector of new candidates
        """
        # Sigmoid layer
        i_t = sigmoid(matrix_calculation(h_t_0, x_t, W_i, U_i, b_i))
        
        # Tanh layer
        Ctilde_t = tanh(matrix_calculation(h_t_0, x_t, W_c, U_c, b_c))
        return i_t, Ctilde_t

    # The output gate controls the extent to which the value in the cell is used to compute the output activation of the LSTM unit
    def output_gate(self, h_t_0, x_t, W_o, U_o, b_o, C_t):
        """Decide what information to output
            The sigmoid layer decides which values to output
            The tanh layer transforms values to range [-1.0, 1.0]
            Multiply by sigmoid layer to scale for what we want to output
        """
        # Sigmoid layer
        o_t = sigmoid(gate(h_t_0, x_t, W_o, U_o, b_o))

        # Tanh layer
        h_t = o_t * tanh_TODO(C_t)
        return o_t, h_t

    # The forget gate controls the extent to which a value remains in the cell
    def forget_gate(self, h_t_0, x_t, W_f, U_f, b_f):
        """Decide what information to throw away from the cell state
            f_t is a vector of continuous values [0.0, 1.0]
            0 is completely forget, 1 completely remember 
        """
        f_t = sigmoid(matrix_calculation(h_t_0, x_t, W_f, U_f, b_f))
        return f_t

    def update_cell_state(self, f_t, C_t_0, i_t, Ctilde_t)
        """Update the cell state with previously calculated value from forget and input gate
            Apply forget vector to old cell state
            Add scaled new candidate values
        """
        C_t = (f_t * C_t_0) + (i_t * Ctilde_t)
        return C_t

    # The activation function of the LSTM gates is often the logistic sigmoid function.
    def sigmoid(self, vector):
        """Scales values of input vector using sigmodal activation to the range [0.0, 1.0]"""
        for idx in range(len(vector)):
            vector[idx] = 1.0 / (1.0 + e^(-vector[idx]))
        return vector

    # Tanh scaling function - hyperbolic tangent activation function
    def tanh(self, vector):
        """Scales the values of the input vector using the hyperbolic tangent function to the range [-1.0, 1.0]"""
        for idx in range(len(vector)):
            x = vector[idx]
            vector[idx] = (e^x - e^-x) / (e^x + e^-x)
        return vector

    # LSTM update function. Feed input, retrieve output using output function.
    def update(self, x_t):
        """Update the cell with new input
            x_t = the new input
        """
        # the output of the previous update
        h_t_0 = self.h_t
        
        # the previous cell content 
        C_t_0 = self.C_t

        # step 1: Decide what information to forget
        f_t = forget_gate(h_t_0, x_t, self.W_f, self.U_f, self.b_f)

        # step 2: Decide what information to put in the cell
        #   2.1 input gate decides which values to update
        #   2.2 tanh layer creates vector of new candidate values
        i_t, Ctilde_t = input_gate(h_t_0, x_t, self.W_i, self.U_i, self.b_i, self.W_c, self.U_c, self.b_c)

        # step 3: Update old cell staet Ct-1 into new cell state Ct
        #   3.1 multiply old state by ft to forget
        #   3.2 add it * C_t, new scaled candidates
        C_t = update_cell_state(f_t, C_t_0, i_t, Ctilde_t)

        # step 4: Decide what the output is
        #   4.1 Run a signmoid layer to decide what part to output
        #   4.2 Put a cell through tanh -> transform to range [-1, 1]
        h_t = output_gate(h_t_0, x_t, self.W_o, self.U_o, self.b_o, C_t)

        # Update cell values, output and the time
        self.h_t = h_t
        self.C_t = C_t
        self.t += 1

    def output(self):
        return self.h_t

    def train(self, training_data):
        pass


def main():
    pass

if __name__ == '__main__':
	main() 