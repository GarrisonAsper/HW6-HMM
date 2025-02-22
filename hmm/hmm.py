import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        

        # Step 0. Handle edge case 
        if len(input_observation_states) == 0:
            return 0.0  # An empty sequence has probability 0
        
        # Step 1. Initialize variables
        T = len(input_observation_states)
        N = len(self.hidden_states)
        alpha = np.zeros((T, N))

        #initialization of forward probability matrix alpha
        for i in range(N):
            alpha[0, i] = self.prior_p[i] * self.emission_p[i, self.observation_states_dict[input_observation_states[0]]]
        
        # Step 2. Calculate probabilities
        #iterate over both time T and number of hidden states N to sum the P(i ->j) * P(emission)
        for t in range(1, T):
            for j in range(N):
                alpha[t, j] = sum(alpha[t-1, i] * self.transition_p[i, j] for i in range(N)) * self.emission_p[j, self.observation_states_dict[input_observation_states[t]]]

        # Step 3. Return final probability 
        #returns sum of the last alpha row
        return sum(alpha[T-1, :])


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        
        # Step 0. Fringe Case handling
            # Handle edge case where input sequence is empty
        if len(decode_observation_states) == 0:
            return []  #empty sequence should return an empty state sequence
        
        # Step 1. Initialize variables
        
        #store probabilities of hidden state at each step 
        T = len(decode_observation_states)  #number of time steps
        N = len(self.hidden_states)  #number of hidden states
        viterbi_table = np.full((T, N), -np.inf)  #T x N matrix initialized with -inf to handle log probabilities to store a sequence of hidden states up until that time T
        backpointer = np.zeros((T, N), dtype=int)  #T X N matrix to store best previous hidden state for backtracking
        
        #initial probabilites in log space: so P(initial_state) * P(emission | initial_state) is actually P(initial_state) + P(emission | initial_state)
        for i in range(N):
            viterbi_table[0, i] = np.log(self.prior_p[i]) + np.log(self.emission_p[i, self.observation_states_dict[decode_observation_states[0]]])
       # Step 2. Calculate Probabilities
        
        for t in range(1, T): #t = 0 is already initialized so we can skip
            for j in range(N):#iterates over all hidden states
                log_probs = [viterbi_table[t-1, i] + np.log(self.transition_p[i, j]) for i in range(N)] #log[P(i transition to j)]
                viterbi_table[t, j] = max(log_probs) + np.log(self.emission_p[j, self.observation_states_dict[decode_observation_states[t]]]) #stores highest log[P(i transition to j)] + log[P(emmision given j)]
                backpointer[t, j] = np.argmax(log_probs) #stores best i that lead to state j

        # Step 3. Traceback 
        best_last_state = np.argmax(viterbi_table[T-1, :]) #gets index of hidden state with highest probability
        best_hidden_state_sequence = [best_last_state]
        
        for t in range(T-1, 0, -1): #iterates through T backwards
            best_hidden_state_sequence.insert(0, backpointer[t, best_hidden_state_sequence[0]])


        # Step 4. Return best hidden state sequence 
                
        return [self.hidden_states_dict[state] for state in best_hidden_state_sequence]