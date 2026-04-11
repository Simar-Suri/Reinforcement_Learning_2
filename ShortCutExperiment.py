# Write your experiments in here! You can use the plotting helper functions from the previous assignment if you want.
from matplotlib import pyplot as plt
import numpy as np
from ShortCutEnvironment import ShortcutEnvironment, WindyShortcutEnvironment
from ShortCutAgents import QLearningAgent, SARSAAgent, ExpectedSARSAAgent, nStepSARSAAgent

def run_repitions(n_rep, agent_type, n_episode,n_actions=4, n_states=144, epsilon=0.1, alpha=0.1, gamma=1.0, env_type = ShortcutEnvironment,n=1):
    agent_returms = []
    if agent_type == nStepSARSAAgent:
        for _ in range(n_rep):
            agent = agent_type(n_actions, n_states, n, epsilon, alpha, gamma, env_type)
            returns = agent.train(n_episode)
            agent_returms.append(returns)
        avg_returns = np.mean(agent_returms, axis=0)  
    
    else:
        for _ in range(n_rep):
            agent = agent_type(n_actions, n_states, epsilon, alpha, gamma, env_type)
            returns = agent.train(n_episode)
            agent_returms.append(returns)
        avg_returns = np.mean(agent_returms, axis=0)  

    return avg_returns

'''agent_007 = run_repitions(1, QLearningAgent, 10000)

plt.plot(agent_007)
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('Q-Learning Learning Agent')
plt.show()

agent_008 = run_repitions(100, QLearningAgent, 1000)

plt.plot(agent_007)
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('Q-Learning Learning Agent')
plt.show()

alpha = [0.01, 0.1, 0.5, 0.9]

for a in alpha:
    returns = run_repitions(100, QLearningAgent, 1000, alpha=a)

    plt.plot(returns, label=f'alpaha:{a}')

plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('Q-Learning with Changing Alpha Values')
plt.legend()
plt.show()

agent_009 = run_repitions(1, SARSAAgent, 10000)

plt.plot(agent_009)
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('SAARSA Agent')
plt.show()

agent_010 = run_repitions(100, SARSAAgent, 1000)

plt.plot(agent_010)
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('SAARSA Agent')
plt.show()

alpha = [0.01, 0.1, 0.5, 0.9]

for a in alpha:
    returns = run_repitions(100, SARSAAgent, 1000, alpha=a)

    plt.plot(returns, label=f'alpaha:{a}')

plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('SAARSA with Changing Alpha Values')
plt.legend()
plt.show()

alpha = [0.01, 0.1, 0.5, 0.9]

for a in alpha:
    returns = run_repitions(100, ExpectedSARSAAgent, 1000, alpha=a)

    plt.plot(returns, label=f'alpaha:{a}')

plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('ExpectedSAARSA with Changing Alpha Values')
plt.legend()
plt.show()'''

n= [1, 2, 5, 10, 25]

for a in n:
    returns = run_repitions(100, nStepSARSAAgent, 1000, n=a)

    plt.plot(returns, label=f'n:{a}')

plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('nStepSAARSA with Changing n Values')
plt.legend()
plt.show()

