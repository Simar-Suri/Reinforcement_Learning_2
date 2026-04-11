# Write your experiments in here! You can use the plotting helper functions from the previous assignment if you want.
from matplotlib import pyplot as plt
import numpy as np
from ShortCutEnvironment import ShortcutEnvironment, WindyShortcutEnvironment
from ShortCutAgents import QLearningAgent, SARSAAgent, ExpectedSARSAAgent, nStepSARSAAgent
from scipy.signal import savgol_filter

class LearningCurvePlot:
    def __init__(self,title=None):
        self.fig,self.ax = plt.subplots()
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Cumulative Reward')
        if title is not None:
            self.ax.set_title(title)

    def add_curve(self,y,label=None):
        ''' y: vector of average reward results
        label: string to appear as label in plot legend '''
        if label is not None:
            self.ax.plot(y,label=label)
            self.ax.legend()
        else:
            self.ax.plot(y)

    def save(self,name='test.png'):
        ''' name: string for filename of saved figure '''
        self.ax.legend()
        self.fig.savefig(name,dpi=300)
        plt.show()

def smooth(y, window=31, poly=1):
    '''
    y: vector to be smoothed
    window: size of the smoothing window '''
    return savgol_filter(y,window,poly)


def run_repitions(n_rep, agent_type, n_episode,n_actions=4,
                  n_states=144, epsilon=0.1, alpha=0.1, gamma=1.0, env_type = ShortcutEnvironment, n_steps=None):
    agent_returns = []
    for _ in range(n_rep):
        if n_steps is not None:
            agent = agent_type(n_actions, n_states, n_steps, epsilon, alpha, gamma, env_type)
        else:
            agent = agent_type(n_actions, n_states, epsilon, alpha, gamma, env_type)
        returns = agent.train(n_episode)
        agent_returns.append(returns)
    avg_returns = np.mean(agent_returns, axis=0)

    return avg_returns

# Part 1
env = ShortcutEnvironment()
agent_1b = QLearningAgent(4, 144, 0.1, 0.1, 1.0, ShortcutEnvironment)
agent_1b.train(10000)
print("Q-learning greedy policy with n_episodes=10000")
env.render_greedy(agent_1b.Q)

agent_008 = run_repitions(100, QLearningAgent, 1000)
plot = LearningCurvePlot(title='Q-Learning Learning Curve (100 repetitions)')
plot.add_curve(agent_008, label='Q-Learning')
plot.save('qlearning_curve.png')

alpha = [0.01, 0.1, 0.5, 0.9]
plot = LearningCurvePlot(title='Q-Learning with Changing Alpha Values')
for a in alpha:
    returns = run_repitions(100, QLearningAgent, 1000, alpha=a)
    plot.add_curve(smooth(returns), label=f'alpha={a}')
plot.save('qlearning_changing_alpha.png')


#Part 2
agent_2b = SARSAAgent(4, 144, 0.1, 0.1, 1.0, ShortcutEnvironment)
agent_2b.train(10000)
print("SARSA greedy policy with n_episodes=10000")
env.render_greedy(agent_2b.Q)

agent_010 = run_repitions(100, SARSAAgent, 1000)
plot = LearningCurvePlot(title='SARSA Learning Curve (100 repetitions)')
plot.add_curve(agent_010, label='SARSA')
plot.save('sarsa_curve.png')

plot = LearningCurvePlot(title='SARSA with Changing Alpha Values')
for a in alpha:
    returns = run_repitions(100, SARSAAgent, 1000, alpha=a)
    plot.add_curve(smooth(returns), label=f'alpha={a}')
plot.save('sarsa_changing_alpha.png')

#Part 3
env_windy = WindyShortcutEnvironment()
agent_3a = QLearningAgent(4, 144, 0.1, 0.1, 1.0, WindyShortcutEnvironment)
agent_3a.train(10000)
print("Windy Q-Learning greedy policy with n_episodes=10000")
env_windy.render_greedy(agent_3a.Q)

agent_3b = SARSAAgent(4, 144, 0.1, 0.1, 1.0, WindyShortcutEnvironment)
agent_3b.train(10000)
print("Windy SARSA greedy policy with n_episodes=10000")
env_windy.render_greedy(agent_3b.Q)

#Part 4
agent_4b = ExpectedSARSAAgent(4, 144, 0.1, 0.1, 1.0, ShortcutEnvironment)
agent_4b.train(10000)
print("Expected SARSA greedy policy with n_episodes=10000")
env.render_greedy(agent_4b.Q)

plot = LearningCurvePlot(title='Expected SARSA with Changing Alpha Values')
for a in alpha:
    returns = run_repitions(100, ExpectedSARSAAgent, 1000, alpha=a)
    plot.add_curve(smooth(returns), label=f'alpha={a}')
plot.save('expected_sarsa_changing_alpha.png')

#Part 5
agent_5b = nStepSARSAAgent(4, 144, 2, 0.1, 0.1, 1.0, ShortcutEnvironment)
agent_5b.train(10000)
print("n-step SARSA greedy policy with n_episodes=10000 (n_steps=2)")
env.render_greedy(agent_5b.Q)


n_values = [1, 2, 5, 10, 25]
plot = LearningCurvePlot(title='n-step SARSA with Changing n_steps Values (alpha=0.1)')
for n in n_values:
    returns = run_repitions(100, nStepSARSAAgent, 1000, alpha=0.1, n_steps=n)
    plot.add_curve(smooth(returns), label=f'n={n}')
    print(f'n={n}: average last 100 episodes = {np.mean(returns[-100:])}')
plot.save('n_step_sarsa_changing_n_steps.png')


#Part 6
plot = LearningCurvePlot(title='Comparison of all algorithms (100 reps, 1000 episodes)')
ql_best  = run_repitions(100, QLearningAgent,1000, alpha=0.1)
sa_best  = run_repitions(100, SARSAAgent,1000, alpha=0.1)
esa_best = run_repitions(100, ExpectedSARSAAgent,1000, alpha=0.9)
nsa_best = run_repitions(100, nStepSARSAAgent,1000, alpha=0.1, n_steps=2)

plot.add_curve(smooth(ql_best),  label='Q-Learning')
plot.add_curve(smooth(sa_best),  label='SARSA')
plot.add_curve(smooth(esa_best), label='Expected SARSA')
plot.add_curve(smooth(nsa_best), label='n-step SARSA (n=5)')
plot.save('comparison.png')

algorithms = ['Q-Learning', 'SARSA', 'Expected SARSA', 'n-step SARSA (n=2)']
results    = [ql_best, sa_best, esa_best, nsa_best]

print(f"\n{'Algorithm':<15} {'Mean last 100 episodes':>15}")
for name, result in zip(algorithms, results):
    print(f"{name:<15} {np.mean(result[-100:]):>15f}")