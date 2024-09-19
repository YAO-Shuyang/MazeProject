from scipy.optimize import minimize
import numpy as np

class HiddenStateModel:
    def __init__(self, alpha=0.25, beta=0.3, a=1.0, r_0=0.5):
        """
        Initialize the Hidden State Model with parameters for increase and decrease functions.
        
        Args:
        - alpha: multiplier for the increase function
        - beta: multiplier for the decrease function
        - a, b: parameters for the increase and decrease functions
        - r_0: initial hidden state (default 0.5)
        """
        self.alpha = alpha  # controls the increase rate
        self.beta = beta    # controls the decrease rate
        self.a = a          # parameter for increase and decrease functions
        #self.b = b          # parameter for increase and decrease functions
        self.r_0 = r_0      # initial hidden state

    def update_hidden_state(self, s_i, r_i):
        """
        Update rule for the hidden state r_i based on the observable state s_i.
        
        Args:
        - s_i: observable state at step i, either 0 or 1
        - r_i: hidden state at step i, a probability in [0, 1]
        
        Returns:
        - r_i+1: updated hidden state for step i+1
        """
        
        if s_i == 1:
            # Increase following 1 - 1 / (a * r_i + b)
            return max(2 - r_i - 2 * np.sqrt(1 / self.a), 0.001)
        else:
            # Decrease following exp(-(r_i/a)^b)
            return r_i * np.exp(-(r_i / self.alpha) ** self.beta)

    def simulate_sequence(self, sequence):
        """
        Simulate the hidden state updates for a given sequence of observable states.
        
        Args:
        - sequence: a list or array of observable states (0 or 1)
        
        Returns:
        - r_values: list of hidden states over the sequence
        """
        r_values = [self.r_0]  # Start with the initial hidden state
        for s_i in sequence:
            r_next = self.update_hidden_state(s_i, r_values[-1])
            r_values.append(r_next)
        return r_values

    def loss_function(self, params, sequences):
        """
        Compute the mean squared error loss over all sequences.
        
        Args:
        - params: A list or array containing [alpha, beta]
        - sequences: A list or array of sequences
        
        Returns:
        - mse: Mean squared error of the fit
        """
        alpha, beta, a = params
        total_error = 0
        num_sequences = len(sequences)

        for sequence in sequences:
            # Simulate with given parameters
            self.alpha = alpha
            self.beta = beta
            self.a = a
            predicted_r_values = self.simulate_sequence(sequence)
    
            # Estimate the observed probability of 1
            observed_prob = np.mean(sequence)  # A rough estimate based on the sequence
            error = np.mean((np.array(predicted_r_values) - observed_prob) ** 2)
            total_error += error
        
        mse = total_error / num_sequences
        return mse

    def fit(self, sequences):
        """
        Fit the model parameters (alpha and beta) to the given data using optimization.
        
        Args:
        - sequences: A list of sequences to fit the model to (each sequence is an array of 0/1 values)
        
        Returns:
        - optimal_alpha: Best-fit alpha value
        - optimal_beta: Best-fit beta value
        """
        initial_params = [self.alpha, self.beta, self.a]
        result = minimize(self.loss_function, initial_params, args=(sequences,), bounds=[(0, 1), (0, 1), (0, 1)])

        # Update model parameters with the optimized values
        self.alpha, self.beta, self.a = result.x
        
        return self.alpha, self.beta, self.a

    def generate_sequence(self, length):
        """
        Generate a single sequence of given length based on the fitted model.
        
        Args:
        - length: the length of the sequence to generate
        
        Returns:
        - sequence: the generated sequence of 0/1 observations
        - r_values: the hidden state values for the generated sequence
        """
        r_values = [self.r_0]  # Start with the initial hidden state
        sequence = [1]
        
        for _ in range(length-1):
            r_i = r_values[-1]
            # Generate s_i based on r_i as a Bernoulli trial
            s_i = np.random.binomial(1, r_i)
            sequence.append(s_i)
            # Update r_i based on the model
            r_next = self.update_hidden_state(s_i, r_i)
            r_values.append(r_next)
        
        return sequence, r_values

    def simulate(self, n_sequences, length=26):
        """
        Generate n sequences of varying lengths based on the fitted model.
        
        Args:
        - n_sequences: the number of sequences to generate
        - length_range: a tuple specifying the range of sequence lengths (default 2 to 26)
        
        Returns:
        - generated_sequences: a list of generated sequences
        """
        generated_sequences = []
        r_values = []
        
        for _ in range(n_sequences):
            # Randomly choose a sequence length within the specified range
            sequence, r_val = self.generate_sequence(length)
            generated_sequences.append(sequence)
            r_values.append(r_val)
        
        return generated_sequences, r_values
    
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mylib.field.tracker_v2 import Tracker2d
    import pickle
    import copy as cp
    
    with open(r"E:\Data\maze_learning\PlotFigures\STAT_CellReg\10227\Maze1-footprint\trace_mdays_conc.pkl", 'rb') as handle:
        trace = pickle.load(handle)

    field_reg = cp.deepcopy(trace['field_reg'][0:, :])
    tracker = Tracker2d(field_reg=field_reg)
    sequences = tracker.convert_to_sequence()
        
    # Create model instance
    model = HiddenStateModel()

    # Fit the model to the sequences
    optimal_alpha, optimal_beta, optimal_a = model.fit(sequences)

    print(optimal_alpha, optimal_beta, optimal_a)
    
    simu_seq, r_val = model.simulate(20000)
    simu_seq = np.vstack(simu_seq)
    from mylib.field.tracker_v2 import Tracker2d

    probs = Tracker2d.recovery_prob2d(simu_seq)
    print("Tracking Done!")

    # plot 3d surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(np.arange(probs.shape[1]), np.arange(probs.shape[0]))
    ax.plot_surface(X, Y, (probs)/2, cmap='viridis', edgecolor='none', alpha=0.8)
    ax.set_xlabel('Inaction')
    ax.set_ylabel('Action')
    ax.view_init(azim=-30, elev=30)
    ax.set_zlim(0, 1)
    plt.show()