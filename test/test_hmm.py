import pytest
from hmm import HiddenMarkovModel
import numpy as np




def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')

    hmm = HiddenMarkovModel(
        mini_hmm['observation_states'],
        mini_hmm['hidden_states'],
        mini_hmm['prior_p'],
        mini_hmm['transition_p'],
        mini_hmm['emission_p']
    )


    #Test Forward Algorithm
    forward_prob = hmm.forward(mini_input['observation_state_sequence'])
    assert np.isclose(forward_prob, 0.03506, atol=1e-5)
    
    #Test Viterbi algorithm
    viterbi_sequence = np.array(hmm.viterbi(mini_input['observation_state_sequence']))
    expected_sequence = np.array(mini_input['best_hidden_state_sequence'], dtype=str)
    assert np.array_equal(viterbi_sequence, expected_sequence)


    #empty observation sequence
    empty_sequence = np.array([])
    empty_forward_prob = hmm.forward(empty_sequence)
    assert empty_forward_prob == 0
    empty_viterbi_sequence = hmm.viterbi(empty_sequence)
    assert empty_viterbi_sequence == []
    
    #single observation sequence
    single_obs_sequence = np.array([mini_input['observation_state_sequence'][0]])
    single_forward_prob = hmm.forward(single_obs_sequence)
    assert single_forward_prob >= 0
    single_viterbi_sequence = hmm.viterbi(single_obs_sequence)
    assert len(single_viterbi_sequence) == 1
   

def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """
    full_hmm = np.load('./data/full_weather_hmm.npz')
    full_input = np.load('./data/full_weather_sequences.npz')
    hmm = HiddenMarkovModel(
        full_hmm['observation_states'],
        full_hmm['hidden_states'],
        full_hmm['prior_p'],
        full_hmm['transition_p'],
        full_hmm['emission_p']
    )

    # Test Forward Algorithm
    forward_prob = hmm.forward(full_input['observation_state_sequence'])
    assert np.isclose(forward_prob, 1.68645e-11, atol=1e-16)
    
    # Test Viterbi Algorithm
    viterbi_sequence = np.array(hmm.viterbi(full_input['observation_state_sequence']))
    expected_sequence = np.array(full_input['best_hidden_state_sequence'], dtype=str)
    assert np.array_equal(viterbi_sequence, expected_sequence)














