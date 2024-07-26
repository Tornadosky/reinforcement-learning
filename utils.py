import numpy as np
import random
import copy

def print_policy(pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4, title='Policy:'):
    print(title)
    arrs = {k:v for k,v in enumerate(action_symbols)}
    for s in range(len(P)):
        a = pi(s)
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), arrs[a].rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")

def print_state_value_function(V, P, n_cols=4, prec=3, title='State-value function:'):
    print(title)
    for s in range(len(P)):
        v = V[s]
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), '{}'.format(np.round(v, prec)).rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")

def print_action_value_function(Q, optimal_Q=None, action_symbols=('<', '>'), prec=3, title='Action-value function:'):
    vf_types=('',) if optimal_Q is None else ('', '*', 'err')
    headers = ['s',] + [' '.join(i) for i in list(itertools.product(vf_types, action_symbols))]
    print(title)
    states = np.arange(len(Q))[..., np.newaxis]
    arr = np.hstack((states, np.round(Q, prec)))
    if not (optimal_Q is None):
        arr = np.hstack((arr, np.round(optimal_Q, prec), np.round(optimal_Q-Q, prec)))
    print(tabulate(arr, headers, tablefmt="fancy_grid"))

def probability_success(env, pi, goal_state, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        while not done and steps < max_steps:
            state, _, done, h = env.step(pi(state))
            steps += 1
        results.append(state == goal_state)
    return np.sum(results)/len(results)

def mean_return(env, pi, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        results.append(0.0)
        while not done and steps < max_steps:
            state, reward, done, _ = env.step(pi(state))
            results[-1] += reward
            steps += 1
    return np.mean(results)

def policy_evaluation(P, pi, gamma=1.0, theta=1e-10):
    nS = len(P.keys())
    V = np.zeros(nS)
    while True:
        prev_V = np.copy(V)
        for s in range(nS):
            v = 0
            for proba, next_s, reward, done in P[s][pi(s)]:
                v += proba * (reward + gamma * prev_V[next_s] * (not done))
            V[s] = v
        if np.max(np.abs(prev_V - V)) < theta:
            break
    return V

def policy_improvement(P, V, gamma=1.0):
    nS, nA = len(P.keys()), len(P[0].keys())
    Q = np.zeros((nS, nA))

    for s in range(nS):
        for a in range(nA):
            for proba, next_s, reward, done in P[s][a]:
                Q[s, a] += proba * (reward + gamma * V[next_s] * (not done))
    
    new_pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return new_pi

def policy_iteration(P, gamma=1.0, theta=1e-10):
    nS, nA = len(P.keys()), len(P[0].keys())
    pi = lambda s: {s:np.random.choice(range(nA)) for s in range(nS)}[s]
    while True:
        old_pi = lambda s: {s:pi(s) for s in range(nS)}[s]
        V = policy_evaluation(P, pi, gamma, theta)
        pi = policy_improvement(P, V, gamma)
        print_policy(pi, P)
        is_equal = True
        for s in range(nS):
            if pi(s) != old_pi(s):
                is_equal = False
                break
        if is_equal:
            break
    return pi, V
