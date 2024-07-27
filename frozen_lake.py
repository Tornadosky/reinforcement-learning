import gym
from utils import policy_evaluation, policy_improvement, policy_iteration, print_state_value_function, print_policy, probability_success, mean_return, value_iteration, render_policy

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="human")

env.reset()
#env.render()

random_pi = lambda s: {
    0:1, 1:3, 2:0, 3:0,
    4:3, 5:2, 6:1, 7:3,
    8:0, 9:2, 10:0, 11:3,
    12:3, 13:1, 14:2, 15:3
}[s]
V = policy_evaluation(env.P, random_pi, gamma=1.0, theta=1e-10)
print_state_value_function(V, env.P, n_cols=4, prec=3, title='State-value function:')

pi, V = policy_iteration(env.P, gamma=0.99)
print_policy(pi, env.P)
print_state_value_function(V, env.P, n_cols=4, prec=3, title='State-value function:')

pi, V = value_iteration(env.P)
print_policy(pi, env.P)

success_rate = probability_success(env, pi, goal_state=15)*100
mean_return = mean_return(env, pi)
print(f'Reaches goal {success_rate:.2f}%. Obtains an average undiscounted return of {mean_return:.4f}.')

# See how the policy performs in the environment
# render_policy(env, random_pi)
