import numpy as np
from ShortCutEnvironment import ShortcutEnvironment, Environment
class QLearningAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((n_states, n_actions))
        # TO DO: Initialize variables if necessary
         
    def select_action(self, state):
        # TO DO: Implement policy
        action = None
        greedy_prob = np.random.rand()
        if greedy_prob <= self.epsilon:
            action = np.random.randint(0, self.n_actions)

        else:
            action = np.argmax(self.Q[state])

        return action
        
    def update(self, state, action, next_state, reward, done): # Augment arguments if necessary
        if done:
            self.Q[state][action] +=  + self.alpha * (reward - self.Q[state][action])
        else: 
            self.Q[state][action] +=  + self.alpha * (reward + (self.gamma * np.max(self.Q[next_state])) - self.Q[state][action])

    
    def train(self, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []
        environment = ShortcutEnvironment()
        for run in range(n_episodes):
            environment.reset()
            run_rewards = []
            while environment.done() is not True:
                prev_state = environment.state()
                action = self.select_action(prev_state)
                reward = environment.step(action)   
                run_rewards.append(reward)     
                self.update(prev_state, action, environment.state(), reward, environment.done())
                
            episode_returns.append(sum(run_rewards))
        return episode_returns


class SARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary
        
    def select_action(self, state):
        # TO DO: Implement policy
        action = None
        return action
        
    def update(self, state, action, reward, done): # Augment arguments if necessary
        # TO DO: Implement SARSA update
        pass

    def train(self, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []
        return episode_returns


class ExpectedSARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary
        
    def select_action(self, state):
        # TO DO: Implement policy
        action = None
        return action
        
    def update(self, state, action, reward, done): # Augment arguments if necessary
        # TO DO: Implement Expected SARSA update
        pass

    def train(self, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []
        return episode_returns    


class nStepSARSAAgent(object):

    def __init__(self, n_actions, n_states, n, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.n = n
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary
        
    def select_action(self, state):
        # TO DO: Implement policy
        action = None
        return action
        
    def update(self, states, actions, rewards, done): # Augment arguments if necessary
        # TO DO: Implement n-step SARSA update
        pass
    
    def train(self, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []
        return episode_returns  
    
    
    