import torch
import torch.nn.functional as F
from torch.optim import Adam
from sac.utils import  soft_update, hard_update
from sac.model import TransferQNetwork, TransferGaussianPolicy, Encoder, QNetwork


class HARDSAC(object):
    def __init__(self, num_inputs, action_space, dynamics_model, dynamics_action_encode, reward_model, reward_action_encode, args):
        torch.autograd.set_detect_anomaly(True)
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.args = args

        self.dynamics_model         = dynamics_model
        self.dynamics_action_encode = dynamics_action_encode
        self.reward_model           = reward_model
        self.reward_action_encode   = reward_action_encode

        self.target_update_interval   = self.args.target_update_interval
        self.automatic_entropy_tuning = self.args.automatic_entropy_tuning

        self.device = torch.device("cuda")

        self.encoder       = Encoder(num_inputs, self.args.hidden_size, self.args.feature_size).to(self.device)
        self.policy        = TransferGaussianPolicy(self.args.feature_size, action_space.shape[0], self.args.hidden_size, action_space).to(self.device)
        self.critic        = TransferQNetwork(num_inputs, action_space.shape[0], self.args.hidden_size).to(device=self.device)
        self.critic_target = TransferQNetwork(num_inputs, action_space.shape[0], self.args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.automatic_entropy_tuning == True:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
            self.log_alpha      = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim    = Adam([self.log_alpha], lr=args.lr)


        self.critic_optim        = Adam(self.critic.parameters(), lr=args.lr)
        self.policy_optim        = Adam(self.policy.parameters(), lr=args.lr)
        self.policy_encode_optim = Adam([{'params': self.encoder.parameters()},
                                         {'params': self.policy.parameters()}], lr=args.lr)
        if args.is_transfer:
            self.encoder_optim = Adam([{'params': self.encoder.parameters()},
                                       {'params': filter(lambda p: p.requires_grad, self.dynamics_model.parameters()),
                                        'lr': 0},
                                       {'params': filter(lambda p: p.requires_grad,
                                                         self.dynamics_action_encode.parameters()), 'lr': 0},
                                       {'params': filter(lambda p: p.requires_grad, self.reward_model.parameters()),
                                        'lr': 0},
                                       {'params': filter(lambda p: p.requires_grad,
                                                         self.reward_action_encode.parameters()), 'lr': 0}], lr=args.lr)

            for param in self.dynamics_model.parameters():
                param.requires_grad = False

            for param in self.dynamics_action_encode.parameters():
                param.requires_grad = False

            for param in self.reward_model.parameters():
                param.requires_grad = False

            for param in self.reward_action_encode.parameters():
                param.requires_grad = False

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        feature = self.encoder(state)
        if eval == False:
            action, _, _ = self.policy.sample(feature)
        else:
            _, _, action = self.policy.sample(feature)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory

        state_batch      = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch     = torch.FloatTensor(action_batch).to(self.device)
        reward_batch     = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch       = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        feature_batch = self.encoder(state_batch)

        with torch.no_grad():
            feature_policy_batch                    = self.encoder(state_batch)
            next_feature_batch                      = self.encoder(next_state_batch)
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_feature_batch)
            qf1_next_target, qf2_next_target        = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target                      = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value                            = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            dynamics_model_label                    = next_feature_batch
            reward_model_label                      = reward_batch

        qf1, qf2 = self.critic(state_batch, action_batch)

        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)

        if self.args.is_transfer:
            with torch.no_grad():
                dynamics_action_feature = self.dynamics_action_encode(action_batch)
                reward_action_feature   = self.reward_action_encode(action_batch)
                dynamics_model_pred     = self.dynamics_model(feature_batch, dynamics_action_feature)
                reward_model_pred       = self.reward_model(feature_batch, reward_action_feature)
            dynamics_model_loss = F.mse_loss(dynamics_model_pred, dynamics_model_label)
            reward_model_loss = F.mse_loss(reward_model_pred, reward_model_label)

            pi, log_pi, _ = self.policy.sample(feature_policy_batch)
            qf1_pi, qf2_pi = self.critic(state_batch, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
            policy_loss += (dynamics_model_loss + reward_model_loss)

            self.policy_optim.zero_grad()
            policy_loss.backward(retain_graph=True)
            self.policy_optim.step()
        else:
            pi, log_pi, _ = self.policy.sample(feature_batch)

            qf1_pi, qf2_pi = self.critic(state_batch, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
            dynamics_model_loss = torch.tensor(0.).to(self.device)
            reward_model_loss = torch.tensor(0.).to(self.device)

            self.policy_optim.zero_grad()
            policy_loss.backward(retain_graph=True)
            self.policy_optim.step()

        self.critic_optim.zero_grad()
        (qf1_loss + qf2_loss).backward()
        self.critic_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item(), dynamics_model_loss.item(), reward_model_loss.item()

    def update_policy(self, memory, batch_size, updates):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory

        state_batch      = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch     = torch.FloatTensor(action_batch).to(self.device)
        reward_batch     = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch       = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            feature_policy_batch                    = self.encoder(state_batch)
            next_feature_batch                      = self.encoder(next_state_batch)
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_feature_batch)
            qf1_next_target, qf2_next_target        = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target                      = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value                            = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch, action_batch)

        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)

        if self.args.is_transfer:
            pi, log_pi, _ = self.policy.sample(feature_policy_batch)
            qf1_pi, qf2_pi = self.critic(state_batch, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
            self.policy_optim.zero_grad()
            policy_loss.backward(retain_graph=True)
            self.policy_optim.step()
        else:
            feature_batch = self.encoder(state_batch)
            pi, log_pi, _ = self.policy.sample(feature_batch)

            qf1_pi, qf2_pi = self.critic(state_batch, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

            self.policy_encode_optim.zero_grad()
            policy_loss.backward(retain_graph=True)
            self.policy_encode_optim.step()

        self.critic_optim.zero_grad()
        (qf1_loss + qf2_loss).backward()
        self.critic_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()


    def update_encoder(self, memory):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)

        feature_batch = self.encoder(state_batch)

        with torch.no_grad():
            next_feature_batch                      = self.encoder(next_state_batch)
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_feature_batch)
            dynamics_model_label                    = next_feature_batch
            reward_model_label                      = reward_batch
        dynamics_action_feature                 = self.dynamics_action_encode(action_batch)
        reward_action_feature                   = self.reward_action_encode(action_batch)
        dynamics_model_pred                     = self.dynamics_model(feature_batch, dynamics_action_feature)
        reward_model_pred                       = self.reward_model(feature_batch, reward_action_feature)

        dynamics_model_loss = F.mse_loss(dynamics_model_pred, dynamics_model_label)
        reward_model_loss   = F.mse_loss(reward_model_pred, reward_model_label)

        print(dynamics_model_loss.item(), reward_model_loss.item())

        self.encoder_optim.zero_grad()
        (dynamics_model_loss + reward_model_loss).backward()
        self.encoder_optim.step()

        return dynamics_model_loss.item(), reward_model_loss.item()