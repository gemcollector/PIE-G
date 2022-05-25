import torch
import torchvision
import os
import numpy as np
import gym
import utils
from copy import deepcopy
from tqdm import tqdm
from arguments import parse_args
from env.wrappers import make_env
from algorithms.factory import make_agent
from video import VideoRecorder
import augmentations
import matplotlib.pyplot as plt

def obs_to_input(obs):
	if isinstance(obs, utils.LazyFrames):
		_obs = np.array(obs)
	else:
		_obs = obs
	_obs = torch.FloatTensor(_obs).cuda()
	_obs = _obs.unsqueeze(0)
	return _obs


def evaluate(env, agent, video, num_episodes, eval_mode, adapt=False, seed=0, domain_name='cartpole', task_name='swingup', algo='svea'):
	episode_rewards = []
	for i in tqdm(range(num_episodes)):
		if adapt:
			ep_agent = deepcopy(agent)
			ep_agent.init_pad_optimizer()
		else:
			ep_agent = agent
		obs = env.reset()
		video.init(enabled=True)
		done = False
		episode_reward = 0
		while not done:
			obs = obs_to_input(obs)
			with torch.no_grad():
				obs = ep_agent['agent'].encoder(obs)
				stddev = 0.1
				dist = ep_agent['agent'].actor(obs, stddev)
				action = (dist.mean).cpu().numpy()[0]

			next_obs, reward, done, _ = env.step(action)
			video.record(env, eval_mode)
			episode_reward += reward
			if adapt:
				ep_agent.update_inverse_dynamics(*augmentations.prepare_pad_batch(obs, next_obs, action))
			obs = next_obs

		episode_rewards.append(episode_reward)


	return np.mean(episode_rewards)


def main(args):
	# Set seed
	utils.set_seed_everywhere(args.seed)
	print('args.seed:', args.seed)
	args.image_size = 84
	args.image_crop_size = 84
	# Initialize environments
	gym.logger.set_level(40)
	env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed+42,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		mode=args.eval_mode,
		intensity=args.distracting_cs_intensity
	)

	# Set working directory
	work_dir = os.path.join(args.log_dir, args.domain_name+'_'+args.task_name, args.algorithm, str(args.seed))
	print('Working directory:', work_dir)
	assert os.path.exists(work_dir), 'specified working directory does not exist'
	model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
	video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
	video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)

	# Check if evaluation has already been run
	if args.eval_mode == 'distracting_cs':
		results_fp = os.path.join(work_dir, args.eval_mode+'_'+str(args.distracting_cs_intensity).replace('.', '_')+'.pt')
	else:
		results_fp = os.path.join(work_dir, args.eval_mode+'.pt')
	assert not os.path.exists(results_fp), f'{args.eval_mode} results already exist for {work_dir}'

	# Prepare agent
	assert torch.cuda.is_available(), 'must have cuda enabled'
	cropped_obs_shape = (3*args.frame_stack, args.image_crop_size, args.image_crop_size)
	print('Observations:', env.observation_space.shape)
	print('Cropped observations:', cropped_obs_shape)

	agent = torch.load('%s/snapshot.pt'%(work_dir), map_location='cuda:0')


	print(f'\nEvaluating {work_dir} for {args.eval_episodes} episodes (mode: {args.eval_mode})')
	reward = evaluate(env, agent, video, args.eval_episodes, args.eval_mode, seed=args.seed, domain_name=args.domain_name, task_name=args.task_name, algo=args.algorithm)
	print('Reward:', int(reward))

	adapt_reward = None
	if args.algorithm == 'pad':
		env = make_env(
			domain_name=args.domain_name,
			task_name=args.task_name,
			seed=args.seed+42,
			episode_length=args.episode_length,
			action_repeat=args.action_repeat,
			mode=args.eval_mode
		)
		adapt_reward = evaluate(env, agent, video, args.eval_episodes, args.eval_mode, adapt=True)
		print('Adapt reward:', int(adapt_reward))



if __name__ == '__main__':
	args = parse_args()
	main(args)
