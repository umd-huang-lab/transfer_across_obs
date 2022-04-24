
import os
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--env', type=str, default="CartPole-v0")
parser.add_argument('--env-name', type=str, default="cartpole")
parser.add_argument('--name', type=str, default="source")
parser.add_argument('--episodes', type=int, default=200)
parser.add_argument('--feature-size', type=int, default=16)
parser.add_argument('--hiddens', type=int, default=32)
parser.add_argument('--head-layers', type=int, default=1)
parser.add_argument('--coeff', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('-target', action="store_true")
parser.add_argument('-no-reg', action="store_true")
parser.add_argument('-detach-next', action="store_true")
parser.add_argument('--load-from', type=str, default="")
args = parser.parse_args()

env = gym.make(args.env).unwrapped

save_path = "data/{}/".format(args.env_name)
os.makedirs(save_path, exist_ok=True)
os.makedirs("learned_models/{}/".format(args.env_name), exist_ok=True)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# plt.ion()

# if gpu is to be used
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



def build_mlp(
        input_size,
        output_size,
        n_layers,
        size,
        activation = nn.ReLU(),
        output_activation = nn.Identity(),
        init_method=None,
        norm=False
):

    layers = []
    in_size = input_size
    for _ in range(n_layers):
        curr_layer = nn.Linear(in_size, size)
        if init_method is not None:
            curr_layer.apply(init_method)
        layers.append(curr_layer)
        layers.append(activation)
        in_size = size

    last_layer = nn.Linear(in_size, output_size)
    if init_method is not None:
        last_layer.apply(init_method)

    layers.append(last_layer)
    layers.append(output_activation)
    if norm:
        layers.append(L2Norm())
        
    return nn.Sequential(*layers)

class L2Norm(nn.Module):
    def forward(self, x):
        return x / x.norm(p=2, dim=1, keepdim=True)

class DQN(nn.Module):

    def __init__(self, inputs, outputs, hiddens, feature_size, head_layers=2):
        super(DQN, self).__init__()

        self.encoder = build_mlp(inputs, feature_size, 2, hiddens, norm=True)
        self.head = build_mlp(feature_size, outputs, head_layers, hiddens)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = self.encoder(x)
        x = self.head(x)
        return x

class ActionDynamicModel(nn.Module):
    '''Only for discrete action space'''
    def __init__(self, feature_size, num_actions, hiddens):
        super().__init__()

        self.h_size = hiddens
        self.num_actions = num_actions
        self.enc_size = feature_size
        
        self.transitions = build_mlp(self.enc_size, self.enc_size*self.num_actions, 0, self.h_size)
        self.rewards = build_mlp(self.enc_size, self.num_actions, 0, self.h_size)
    
    def forward(self, encoding, actions):
        dist_actions = actions.flatten() #torch.LongTensor(actions).to(device)
        predict_next = self.transitions(encoding)
        predict_reward = self.rewards(encoding)
        
        ind_starts = dist_actions * self.enc_size
        ind_ends = ind_starts + self.enc_size
        indices = torch.stack([torch.arange(ind_starts[i], ind_ends[i]) for i in range(ind_starts.size()[0])]).to(device)
        predict_next = predict_next.gather(1, indices)
        predict_reward = predict_reward.gather(1, dist_actions.view(-1,1))
        
        return predict_next, predict_reward

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


n_actions = env.action_space.n
state_size = env.observation_space.shape[0]
policy_net = DQN(state_size, n_actions, hiddens=args.hiddens, feature_size=args.feature_size, 
                head_layers=args.head_layers).to(device)
target_net = DQN(state_size, n_actions, hiddens=args.hiddens, feature_size=args.feature_size, 
                head_layers=args.head_layers).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
print(policy_net)

dynamic_model = ActionDynamicModel(feature_size=args.feature_size, num_actions=n_actions, hiddens=64).to(device)
print(dynamic_model)

optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
if args.target:
    dynamic_model.load_state_dict(torch.load(args.load_from)["dynamics"])
    print("loaded from", args.load_from)
else:   
    model_optimizer = optim.Adam(dynamic_model.parameters(), lr=args.lr)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())



def model_loss(state_batch, next_batch, action_batch, reward_batch, target=False):
    criterion = nn.SmoothL1Loss()
    if target:
        encodings = target_net.encoder(state_batch).detach()
        next_encodings = target_net.encoder(next_batch).detach()
    else:
        encodings = policy_net.encoder(state_batch)
        next_encodings = policy_net.encoder(next_batch)
        if args.detach_next:
            next_encodings = next_encodings.detach()
    predict_next, predict_reward = dynamic_model(encodings, action_batch)
    model_loss = criterion(predict_next, next_encodings) + criterion(predict_reward.flatten(), reward_batch)

    return model_loss

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward).float()

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # optimize dynamics model
    if not args.target:
        loss_model_target = model_loss(state_batch[non_final_mask], non_final_next_states, action_batch[non_final_mask], reward_batch[non_final_mask], target=True)
        # print(loss_model_target)
        # print(loss_model_target.type())
        model_optimizer.zero_grad()
        loss_model_target.backward()
        for param in dynamic_model.parameters():
            param.grad.data.clamp_(-1, 1)
        model_optimizer.step()
    else:
        loss_model_target = torch.Tensor([0])
    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    if not args.no_reg:
        loss_model = model_loss(state_batch[non_final_mask], non_final_next_states, action_batch[non_final_mask], reward_batch[non_final_mask])
        loss += args.coeff * loss_model
    else:
        loss_model = torch.Tensor([0])
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss_model.item(), loss_model_target.item()


num_episodes = args.episodes
total_rewards = []
mean_loss = []
for i_episode in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()
    state = torch.tensor([state]).float().to(device)
    eps_reward = 0
    for t in count():
        # Select and perform an action
        action = select_action(state)
        next, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        if not done:
            next_state = torch.tensor([next]).float().to(device)
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)
        eps_reward += reward.item()
        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        mloss = optimize_model() 
        if done or t > 200:
            episode_durations.append(t + 1)
            # plot_durations()
            break
    print("episode", i_episode, "reward", eps_reward, "loss", mloss)
    total_rewards.append(eps_reward)
    mean_loss.append(mloss)
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
plt.plot(total_rewards)
plt.savefig("data/{}/{}.png".format(args.env_name, args.name), format="png")
plt.close()
with open("data/{}/{}.txt".format(args.env_name, args.name), "w") as f:
    for i, reward in enumerate(total_rewards):
        f.write("Episode: {}, Reward: {}\n".format(i, reward))

torch.save({
            "dynamics": dynamic_model.state_dict(), 
            "encoder": policy_net.encoder.state_dict(),
            "head": policy_net.head.state_dict()
        },
        "learned_models/{}/{}.pt".format(args.env_name, args.name)
    )
