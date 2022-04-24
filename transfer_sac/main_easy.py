import argparse
import gym
import torch
import numpy as np

import logging
import itertools

from sac.replay_memory import ReplayMemory
from sac.sac_easy import EASYSAC


def readParser():
    parser = argparse.ArgumentParser(description='Source_SAC')
    parser.add_argument('--env_name', default="Hopper-v2",
                        help='Mujoco Gym environment (default: Hopper-v2)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')

    parser.add_argument('--use_decay', type=bool, default=False, metavar='G',
                        help='discount factor for reward (default: 0.99)')

    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--feature_size', type=int, default=256, metavar='N',
                        help='feature size (default: 256)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')

    parser.add_argument('--reward_size', type=int, default=1, metavar='E',
                        help='environment reward size')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')

    parser.add_argument('--num_epoch', type=int, default=1000, metavar='A',
                        help='total number of epochs')
    parser.add_argument('--train_every_n_steps', type=int, default=1, metavar='A',
                        help='frequency of training policy')
    parser.add_argument('--policy_train_batch_size', type=int, default=256, metavar='A',
                        help='batch size for training policy')
    parser.add_argument('--init_exploration_steps', type=int, default=1000, metavar='A',
                        help='exploration steps initially')
    parser.add_argument('--cuda', default=True, action="store_true",
                        help='run on CUDA (default: True)')
    parser.add_argument('--model_dir', default='./model_file/',
                        help='your model save path')
    parser.add_argument('--model_name', default='model.pt',
                        help='your model save path')
    parser.add_argument('--input_type', default='state',
                        help='input type can be state or pixels')
    parser.add_argument('--is_transfer',type=bool , default=False,
                        help='only effective when the input type is pixel')
    parser.add_argument('--is_model_based', type=bool, default=False,
                        help='only effective when the input type is state')
    parser.add_argument('--exp_log_name', default='exp_walker_0.txt')
    parser.add_argument('--updates_per_step', default=1)


    return parser.parse_args()



def train(args, env, env_test, agent, env_pool, logger):
    device = torch.device("cuda")
    total_numsteps = 0
    updates = 0
    print("--------------------Start SAC--------------------")
    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        while not done:
            if args.init_exploration_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(env_pool) > args.policy_train_batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    env_state, env_action, env_reward, env_next_state, env_done = env_pool.sample(int(args.policy_train_batch_size))
                    batch_state, batch_action, batch_reward, batch_next_state, batch_done = env_state, env_action, env_reward, env_next_state, env_done

                    batch_reward, batch_done = np.squeeze(batch_reward), np.squeeze(batch_done)
                    batch_done = (~batch_done).astype(int)
                    agent.update_parameters((batch_state, batch_action, batch_reward, batch_next_state, batch_done), args.policy_train_batch_size, updates)
                    updates += 1

            next_state, reward, done, _ = env.step(action)  # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            env_pool.push(state, action, reward, next_state, done)  # Append transition to memory

            state = next_state

            if total_numsteps % 1000 == 0:
                length = 0
                test_state = env_test.reset()
                sum_reward = 0
                test_done = False
                while not test_done and length < 1000:
                    test_action = agent.select_action(test_state, eval=True)
                    test_next_state, test_reward, test_done, _ = env_test.step(test_action)
                    sum_reward += test_reward
                    test_state = test_next_state
                    length += 1

                print("----------------------------------------")
                print("Total Steps: {}, Test Reward: {}".format(total_numsteps, str(sum_reward)))
                logger.info("Total Steps: " + str(total_numsteps) + " " + "Test Length: " + str(length) + " " + "Test Reward: " + str(sum_reward))

                print("----------------------------------------")

            if total_numsteps % 50000 == 0:
                print("--------------------Start Train Model--------------------")

                agent.train_model(env_pool)
                torch.save({'Dynamics': agent.dynamics_model.state_dict(),
                            'Dynamics_action_encode': agent.dynamics_action_encoder.state_dict(),
                            'Reward': agent.reward_model.state_dict(),
                            'Reward_action_encode': agent.reward_action_encoder.state_dict()
                            }, args.model_dir + '{}-model.pt'.format(args.env_name))

        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps,episode_steps,round(episode_reward, 2)))


def main(args=None):
    if args is None:
        args = readParser()

    # Initial environment
    env = gym.make(args.env_name)
    env_test = gym.make(args.env_name)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)
    env_test.seed(args.seed)

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(args.exp_log_name)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Intial agent
    agent = EASYSAC(num_inputs=env.observation_space.shape[0], action_space=env.action_space, args=args)

    # Initial pool for env
    env_pool = ReplayMemory(args.replay_size)

    print("-------------------Train Policy---------------------")
    train(args, env, env_test, agent, env_pool, logger)

if __name__ == '__main__':
    main()
