# Difference LSTM to ESN --> LSTM is desgined to deal with the vanishing gradient problem, encountered when training RNN's. 
# Relative insensitivity to gap length is an advantage of LSTM over RNNs, hidden Markov models and other sequence learning methods in numerous applications.

class LSTM():

    def __init__(self):
        pass



# Memory part, the cell is responsible for keeping track of the dependencies between the elements in the input sequence. 
class Cell():
    pass

# Regulators -- aka gates -- of the flow of information inside the LSTM unit.


def gate(h, x, W, U, b):
    weighted_output = W * h
    weighted_input = U * x
    output = sigmoid(weighted_output + weighted_input + b)
    return output
# TODO merge top and bottom function
def tanh(h, x, W, U, b):
    weighted_output = W * h
    weighted_input = U * x
    output = tanh_TODO(weighted_output + weighted_input + b)
    return output

# The input gate controls the extent to which a new value flows into the cell
def input_gate(h_t_0, x_t, W_i, U_i, b_i):
    i_t = gate(h_t_0, x_t, W_i, U_i, b_i)
    Ctilde_t = tanh(h_t_0, x_t, W_i, U_i, b_i)
    return i_t, Ctilde_t

# The output gate controls the extent to which the value in the cell is used to compute the output activation of the LSTM unit
def output_gate(h_t_0, x_t, W_o, U_o, b_o, C_t):
    o_t = gate(h_t_0, x_t, W_o, U_o, b_o)
    h_t = o_t * tanh_TODO(C_t)
    return o_t, h_t

# The forget gate controls the extent to which a value remains in the cell
def forget_gate(h_t_0, x_t, W_f, U_f, b_f):
    f_t = gate(h_t_0, x_t, W_f, U_f, b_f)
    return f_t

def update_cell_state(f_t, C_t_0, i_t, Ctilde_t)
    C_t = f_t * C_t_0 + i_t * Ctilde_t
    return C_t

# The activation function of the LSTM gates is often the logistic sigmoid function.
def sigmoid(matrix):
    pass

# Place inside LSTM later, for now build it here
def update():
    # step 1: Decide what information to forget
    f_t = forget_gate()

    # step 2: Decide what information to put in the cell
    #   2.1 input gate decides which values to update
    #   2.2 tanh layer creates vector of new candidate values
     i_t, Ctilde_t = input_gate()

    # step 3: Update old cell staet Ct-1 into new cell state Ct
    #   3.1 multiply old state by ft to forget
    #   3.2 add it * C_t, new scaled candidates
    C_t = update_cell_state()

    # step 4: Decide what the output is
    #   4.1 Run a signmoid layer to decide what part to output
    #   4.2 Put a cell through tanh -> transform to range [-1, 1]
    o_t, h_t = output_gate()


def main():
    pass

if __name__ == '__main__':
	main()