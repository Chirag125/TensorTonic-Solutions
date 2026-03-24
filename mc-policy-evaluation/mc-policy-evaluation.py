import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    V = np.zeros(n_states)
    returns_sum = np.zeros(n_states)
    returns_count = np.zeros(n_states)

    for episode in episodes:
        G = 0
       
        visited_states = set()
        for t in range(len(episode) - 1, -1, -1):
            state, reward = episode[t]
            G = reward + gamma * G
            
            is_first_visit = True
            for prev_t in range(t):
                if episode[prev_t][0] == state:
                    is_first_visit = False
                    break
            
            if is_first_visit:
                returns_sum[state] += G
                returns_count[state] += 1
                V[state] = returns_sum[state] / returns_count[state]

    return V
