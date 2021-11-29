import numpy as np 

import random 

class Agent: 

  
    def __init__(self, Q, mode="test_mode"): 

        self.Q = Q 
        self.mode = mode 
        self.n_actions = 6 

        self.episode = [] 
        self.k = 1


    def select_action(self, state): 
        if self.mode == 'q_learning':
            eps_q = 0.01 
            if random.random() > eps_q: 
                return np.argmax(self.Q[state])  
            else:  
                return (random.choice(np.arange(self.n_actions))) 

        elif self.mode == 'mc_control': 
            eps_m = 1/self.k
            if random.random() > eps_m: 
                return np.argmax(self.Q[state])  
            else:  
                return (random.choice(np.arange(self.n_actions))) 
        elif self.mode =='test_mode':
            eps_q = 0.01 
            if random.random() > eps_q: 
                return np.argmax(self.Q[state])  
            else:  
                return (random.choice(np.arange(self.n_actions))) 




    def step(self, state, action, reward, next_state, done): 
       if self.mode == 'q_learning': 
            new_value_q_learning = self.Q_learning(state, action, reward, next_state, done) 
            self.Q[state][action] = new_value_q_learning 
       elif self.mode == 'mc_control': 
           if not done: 
               self.episode.append((state, action, reward)) 
           else:  
               a = self.update() 
               self.Q = a
               self.episode = []
               self.k += 0.001


    def Q_learning(self, state, action, reward, next_state, done): 
        gamma = 0.90 
        alpha = 0.01 

        # new state = immeditate reward + gamma(value_fuction of next_state- current value function) 
        current_value = self.Q[state][action] 
        next_value_max = np.max(self.Q[next_state]) 
        new_value =  current_value + alpha * (reward + gamma * next_value_max - current_value) 
        return new_value 

  

    # for every episode, agent updates q function of visited states 

    def update(self): 
        G = 0 
        gamma = 0.9
        alpha = 0.05
        for state, action, reward in reversed(self.episode): 
            G = reward + gamma * G  
            current_value = self.Q[state][action] 
            self.Q[state][action] = current_value +  alpha * (G - current_value) 
        return self.Q 

 