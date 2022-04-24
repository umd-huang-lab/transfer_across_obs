# Transfer RL across Observation Feature Spaces via Model-Based Regularization
This is the SAC code for our MuJoCo experiments reported in Figure 4 and 5.

The code is based on [an open-sourced pytorch SAC implementation](https://github.com/pranz24/pytorch-soft-actor-critic).

To use the target tasks in our paper, please first follow the steps in file *envs/test_env_setup.txt* to install the target environments.



## Getting started ###
- Train the dynamics model and the reward model on the source task:
```bash
CUDA_VISIBLE_DEVICES=0 python main_easy.py --env_name 'HalfCheetah-v3' --exp_log_name exp_source_halfcheetah_0.txt
```
- Train SAC on the target task:
```bash
CUDA_VISIBLE_DEVICES=0 python main_hard.py --env_name 'HalfCheetahTest-v3' --exp_log_name exp_target_halfcheetah_0.txt
```
- Train SAC with transferred dynamics models on the target task:
```bash
CUDA_VISIBLE_DEVICES=2 python main_hard.py --env_name 'HalfCheetahTest-v3' --exp_log_name exp_transfer_halfcheetah_0.txt --is_transfer True --model_name HalfCheetah-v3-model.pt
```
