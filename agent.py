import torch
import random, numpy as np
from pathlib import Path

from neural import MarioNet
from collections import deque




class Mario:
    def __init__(self, state_dim, action_dim, save_dir, checkpoint=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=100000)
        self.batch_size = 32

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.gamma = 0.9

        self.curr_step = 0
        self.burnin = 1e5  # min. experiences before training
        self.learn_every = 3   # no. of experiences between updates to Q_online
        self.sync_every = 1e4   # no. of experiences between Q_target & Q_online sync

        self.save_every = 10000  # no. of experiences between saving Mario Net
        self.save_dir = save_dir

        self.use_cuda = torch.backends.mps.is_available() or torch.cuda.is_available()


        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            device = torch.device(self._get_device())
            self.net = self.net.to(device=device)
            print(device)
            
        if checkpoint:
            self.load(checkpoint)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

    # 定义函数判断使用torch的不同后端，cuda|mps|cpu
    def _get_device(self):
        dev="cpu"
        # if self.use_cuda:
        #     if torch.backends.mps.is_available():
        #         dev= "mps"
        #     elif torch.cuda.is_available():
        #         dev= "cuda"
        # # print("Using devcie ======================= "+dev)
        #
        return dev
        
    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state(LazyFrame): A single observation of the current state, dimension is (state_dim)
        Outputs:
        action_idx (int): An integer representing which action Mario will perform
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state_tensor = torch.from_numpy(np.array(state)).float()
            state = state_tensor.to(device=self._get_device()) if self.use_cuda else state_tensor
            state = state.unsqueeze(0)
            action_values = self.net(state, model='online')
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def _to_tensor(self,data, dtype=torch.float32):
        device = self._get_device()
        data = np.array(data)
        tensor = torch.from_numpy(data)
        if dtype is not None:
            tensor = tensor.type(dtype)
        if device is not None:
            tensor = tensor.to(device)
        return tensor
    
    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        # Use numpy to opt// old impl
        
        # state = torch.FloatTensor(state).to(device= "cuda") if self.use_cuda else torch.FloatTensor(state)
        # next_state = torch.FloatTensor(next_state).to(device= "cuda")if self.use_cuda else torch.FloatTensor(next_state)
        # action = torch.LongTensor([action]).to(device= "cuda")if self.use_cuda else torch.LongTensor([action])
        # reward = torch.FloatTensor([reward]).to(device= "cuda")if self.use_cuda else torch.DoubleTensor([reward])
        # done = torch.BoolTensor([done]).to(device= "cuda")if self.use_cuda else torch.BoolTensor([done])

        

        state = self._to_tensor(state)
        next_state = self._to_tensor(next_state )
        action = self._to_tensor(action, dtype=torch.long)
        reward = self._to_tensor(reward)
        done = self._to_tensor(done)
 
        # state_tensor = torch.from_numpy(np.array(state))
        # state_tensor= state_tensor.float()
        # state = state_tensor.to(device="mps") if self.use_cuda else state_tensor
        # next_state = np.array(next_state)
        # next_state = torch.from_numpy(next_state).float().to(device="mps") if self.use_cuda else torch.from_numpy(next_state)
        #
        # action = np.array(action)
        # action = torch.from_numpy(action).to(device="mps") if self.use_cuda else torch.from_numpy(action)
        #
        # reward = np.array(reward)
        # reward = torch.from_numpy(reward).float().to(device="mps") if self.use_cuda else reward
        #
        # done = np.array(done)
        # done = torch.from_numpy(done).to(device="mps") if self.use_cuda else torch.from_numpy(done)
        
        self.memory.append( (state, next_state, action, reward, done,) )


    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()


    def td_estimate(self, state, action):
        current_Q = self.net(state, model='online')[np.arange(0, self.batch_size), action] # Q_online(s,a)
        return current_Q


    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model='online')
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model='target')[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()


    def update_Q_online(self, td_estimate, td_target) :
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())


    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)


    def save(self):
        save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(
            dict(
                model=self.net.state_dict(),
                exploration_rate=self.exploration_rate
            ),
            save_path
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")


    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=(self._get_device()))
        # ckp = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')

        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.net.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate
