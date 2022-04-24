import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

def weights_init_(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        return x1, x2



class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob = log_prob - torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class Encoder(nn.Module):
    def __init__(self, num_inputs, hidden_dim, num_feature):
        super(Encoder, self).__init__()

        self.encode1 = nn.Linear(num_inputs, hidden_dim)
        self.encode2 = nn.Linear(hidden_dim, num_feature)

        self.apply(weights_init_)

    def forward(self, state):
        e1 = F.relu(self.encode1(state))
        feature = F.relu(self.encode2(e1))

        return feature

class DynamicsActionEncoder(nn.Module):
    def __init__(self, num_actions, hidden_dim, num_feature):
        super(DynamicsActionEncoder, self).__init__()

        self.encode1 = nn.Linear(num_actions, hidden_dim)
        self.encode2 = nn.Linear(hidden_dim, num_feature)

        self.apply(weights_init_)

    def forward(self, action):
        e1 = F.relu(self.encode1(action))
        feature = F.relu(self.encode2(e1))

        return feature

class RewardActionEncoder(nn.Module):
    def __init__(self, num_actions, hidden_dim, num_feature):
        super(RewardActionEncoder, self).__init__()

        self.encode1 = nn.Linear(num_actions, hidden_dim)
        self.encode2 = nn.Linear(hidden_dim, num_feature)

        self.apply(weights_init_)

    def forward(self, action):
        e1 = F.relu(self.encode1(action))
        feature = F.relu(self.encode2(e1))

        return feature

class TransferQNetwork(nn.Module):
    def __init__(self, num_feature, num_actions, hidden_dim):
        super(TransferQNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_feature + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_feature + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, feature, action):
        xu = torch.cat([feature, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        return x1, x2


class TransferGaussianPolicy(nn.Module):
    def __init__(self, num_feature, num_actions, hidden_dim, action_space=None):
        super(TransferGaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_feature, hidden_dim)
        # self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        # self.mean_linear = nn.Linear(hidden_dim, num_actions)
        # self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.mean_linear = nn.Linear(num_feature, num_actions)
        self.log_std_linear = nn.Linear(num_feature, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, feature):
        x = feature
        # x = F.relu(self.linear1(feature))
        # x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, feature):
        mean, log_std = self.forward(feature)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob = log_prob - torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(TransferGaussianPolicy, self).to(device)


class PixelEncoder(nn.Module):
    def __init__(self, input_channel, num_feature, conv_hidden_dim=32):
        super(PixelEncoder, self).__init__()

        self.input_channel = input_channel
        self.conv_hidden_dim = conv_hidden_dim
        self.num_feature = num_feature

        self.conv1 = nn.Conv2d(in_channels=self.input_channel, out_channels=self.conv_hidden_dim,
                               kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(in_channels=self.conv_hidden_dim, out_channels=self.conv_hidden_dim * 2,
                               kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=self.conv_hidden_dim * 2, out_channels=self.conv_hidden_dim * 4,
                               kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(in_channels=self.conv_hidden_dim * 4, out_channels=self.conv_hidden_dim * 8,
                               kernel_size=4, stride=2)
        self.conv5 = nn.Conv2d(in_channels=self.conv_hidden_dim * 8, out_channels=self.conv_hidden_dim * 8,
                               kernel_size=4, stride=2)
        self.conv6 = nn.Conv2d(in_channels=self.conv_hidden_dim * 8, out_channels=self.conv_hidden_dim * 8,
                               kernel_size=4, stride=2)
        self.conv7 = nn.Conv2d(in_channels=self.conv_hidden_dim * 8, out_channels=self.conv_hidden_dim * 8,
                               kernel_size=4, stride=2)
        self.latent_linear = nn.Linear(256, self.num_feature)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        self.apply(weights_init_)

    def forward(self, obs):
        obs = obs / 255
        x = self.relu(self.conv1(obs))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.flatten(x)
        latent = self.latent_linear(x)

        return latent

class PixelDecoder(nn.Module):
    def __init__(self, output_channel, num_feature, conv_hidden_dim=32):
        super(PixelDecoder, self).__init__()

        self.output_channel = output_channel
        self.conv_hidden_dim = conv_hidden_dim
        self.num_feature = num_feature

        self.convtrans1 = nn.ConvTranspose2d()




class PixelPolicyNetwork(nn.Module):
    def __init__(self, num_feature: int, hidden_dim: int, action_dim: int, action_space=None):
        super(PixelPolicyNetwork, self).__init__()

        self.num_feature = num_feature
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # Gaussian Policy architecture
        self.linear1 = nn.Linear(self.num_feature, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.mean_linear = nn.Linear(self.hidden_dim, self.action_dim)
        self.log_std_linear = nn.Linear(self.hidden_dim, self.action_dim)

        self.relu = nn.ReLU()

        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

        self.apply(weights_init_)


    def forward(self, latent):
        x = self.relu(self.linear1(latent))
        x = self.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std


    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob = log_prob - torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(PixelPolicyNetwork, self).to(device)


class PixelQNetwork(nn.Module):
    def __init__(self, num_feature: int, hidden_dim: int, action_dim: int):
        super(PixelQNetwork, self).__init__()
        self.num_feature = num_feature
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.relu = nn.ReLU()

        # Q1 architecture
        self.linear1 = nn.Linear(self.num_feature + self.action_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear3 = nn.Linear(self.hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(self.num_feature + self.action_dim, self.hidden_dim)
        self.linear5 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear6 = nn.Linear(self.hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, latent, action):
        xu = torch.cat([latent, action], 1)

        x1 = self.relu(self.linear1(xu))
        x1 = self.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = self.relu(self.linear4(xu))
        x2 = self.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        return x1, x2