import torch
import torch.nn.functional as F
from torch.optim import Adam
from sac.utils import  soft_update, hard_update
from sac.model import TransferQNetwork, TransferGaussianPolicy, Encoder, DynamicsActionEncoder, RewardActionEncoder
from sac.dynamics_model import DynamicsModel, RewardModel
import numpy as np
import itertools

class EASYSAC(object):
    def __init__(self, num_inputs, action_space, args):
        torch.autograd.set_detect_anomaly(True)
        self.gamma = args.gamma
        self.tau   = args.tau
        self.alpha = args.alpha

        self.target_update_interval   = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda")

        self.encoder                 = Encoder(num_inputs, args.hidden_size, args.feature_size).to(self.device)
        self.dynamics_action_encoder = DynamicsActionEncoder(action_space.shape[0], args.hidden_size, args.feature_size)
        self.reward_action_encoder   = DynamicsActionEncoder(action_space.shape[0], args.hidden_size, args.feature_size)

        self.dynamics_model          = DynamicsModel(args.feature_size, args.hidden_size, use_decay=args.use_decay)
        self.reward_model            = RewardModel(args.feature_size, args.reward_size, args.hidden_size, use_decay=args.use_decay)

        self.policy                  = TransferGaussianPolicy(args.feature_size, action_space.shape[0], args.hidden_size, action_space).to(self.device)
        self.critic                  = TransferQNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_target           = TransferQNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.automatic_entropy_tuning == True:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

        self.dynamics_model_optim = Adam([{'params': self.dynamics_model.parameters()},
                                          {'params': self.dynamics_action_encoder.parameters()}], lr=args.lr)
        self.reward_model_optim   = Adam([{'params': self.reward_model.parameters()},
                                          {'params': self.reward_action_encoder.parameters()}], lr=args.lr)
        self.policy_optim         = Adam([{'params': self.encoder.parameters()},
                                          {'params': self.policy.parameters()}], lr=args.lr)
        self.critic_optim         = Adam(self.critic.parameters(), lr=args.lr)


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

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        feature_batch = self.encoder(state_batch)

        with torch.no_grad():
            next_feature_batch = self.encoder(next_state_batch)
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_feature_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch, action_batch)

        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)

        pi, log_pi, _ = self.policy.sample(feature_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        loss = qf1_loss + qf2_loss + policy_loss

        self.policy_optim.zero_grad()
        policy_loss.backward()
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

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), loss.item(), alpha_loss.item(), alpha_tlogs.item()


    def train_model(self, env_pool, batch_size=256, holdout_ratio=0.2, max_epochs_since_update=5):

        state, action, reward, next_state, done = env_pool.sample(len(env_pool))

        self._max_epochs_since_update = max_epochs_since_update
        self._epochs_since_update = 0
        self._state = {}
        self._snapshots = (None, 1e10, 1e10)

        num_holdout = int(state.shape[0] * holdout_ratio)
        permutation = np.random.permutation(state.shape[0])
        state, action, reward, next_state, done = state[permutation], action[permutation], reward[permutation], next_state[permutation], done[permutation]

        train_state, train_action, train_reward, train_next_state, train_done = \
            state[num_holdout:], action[num_holdout:], reward[num_holdout:], next_state[num_holdout:], done[num_holdout:]
        holdout_state, holdout_action, holdout_reward, holdout_next_state, holdout_done = \
            state[num_holdout:], action[num_holdout:], reward[num_holdout:], next_state[num_holdout:], done[num_holdout:]

        holdout_state = torch.from_numpy(holdout_state).float().to(self.device)
        holdout_action = torch.from_numpy(holdout_action).float().to(self.device)
        holdout_reward = torch.from_numpy(holdout_reward).float().to(self.device).unsqueeze(1)
        holdout_next_state = torch.from_numpy(holdout_next_state).float().to(self.device)

        print('-------------------------Start Train-----------------------------')

        for epoch in itertools.count():
            train_idx = np.random.permutation(train_state.shape[0])
            for start_pos in range(0, train_state.shape[0], batch_size):
                idx = train_idx[start_pos: start_pos + batch_size]
                train_states = torch.from_numpy(train_state[idx]).float().to(self.device)
                train_actions = torch.from_numpy(train_action[idx]).float().to(self.device)
                train_rewards = torch.from_numpy(train_reward[idx]).float().to(self.device).unsqueeze(1)
                train_next_states = torch.from_numpy(train_next_state[idx]).float().to(self.device)

                with torch.no_grad():
                    feature_state_train = self.encoder(train_states)
                    feature_next_state_train = self.encoder(train_next_states)

                feature_dynamics_action = self.dynamics_action_encoder(train_actions)
                feature_reward_action   = self.reward_action_encoder(train_actions)

                losses = []
                state_pred  = self.dynamics_model(feature_state_train, feature_dynamics_action)
                state_label = feature_next_state_train
                state_loss  = F.mse_loss(state_pred, state_label)

                reward_pred  = self.reward_model(feature_state_train, feature_reward_action)
                reward_label = train_rewards
                reward_loss  = F.mse_loss(reward_pred, reward_label)

                self.dynamics_model_optim.zero_grad()
                state_loss.backward()
                self.dynamics_model_optim.step()

                self.reward_model_optim.zero_grad()
                reward_loss.backward()
                self.reward_model_optim.step()

                losses.append(state_loss)

            with torch.no_grad():
                feature_state_houldout          = self.encoder(holdout_state)
                feature_next_state_holdout      = self.encoder(holdout_next_state)
                feature_dynamics_action_holdout = self.dynamics_action_encoder(holdout_action)
                feature_reward_action_holdout   = self.reward_action_encoder(holdout_action)

                holdout_state_pred              = self.dynamics_model(feature_state_houldout, feature_dynamics_action_holdout)
                holdout_state_label             = feature_next_state_holdout
                holdout_state_mse_losses        = F.mse_loss(holdout_state_pred, holdout_state_label)
                holdout_state_mse_losses        = holdout_state_mse_losses.detach().cpu().numpy()

                holdout_reward_pred             = self.reward_model(feature_state_houldout, feature_reward_action_holdout)
                holdout_reward_label            = holdout_reward
                holdout_reward_mse_losses       = F.mse_loss(holdout_reward_pred, holdout_reward_label)
                holdout_reward_mse_losses       = holdout_reward_mse_losses.detach().cpu().numpy()

                break_train                     = self._save_best(epoch, holdout_state_mse_losses, holdout_reward_mse_losses)


                if break_train:
                    break
            print('epoch: {}, holdout state mse losses: {}, holdout reward mse losses: {}'.format(epoch, holdout_state_mse_losses, holdout_reward_mse_losses))

    def _save_best(self, epoch, holdout_state_losses, holdout_reward_losses):
        updated = False
        current_state_loss = holdout_state_losses
        current_reward_loss = holdout_reward_losses
        _, best_state, best_reward = self._snapshots
        improvement_state = (best_state - current_state_loss) / best_state
        improvement_reward = (best_reward - current_reward_loss) / best_reward

        if improvement_state > 0.01 or improvement_reward > 0.01:
            self._snapshots = (epoch, current_state_loss, current_reward_loss)
            updated = True

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False