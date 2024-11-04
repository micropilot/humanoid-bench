import os
import mujoco.viewer
import numpy as np
import time
from .flax_to_torch import TorchModel, TorchPolicy
from humanoid_bench.mjx.envs.cpu_env import HumanoidNumpyEnv
import tqdm
import cv2
from .video_utils import save_numpy_as_video, make_grid_video_from_numpy
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.animation as animation

images = []

# Function to update the animation
def update(img):
    plt.clf()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.imshow(img)
    plt.axis('off')


def main(args):
    if args.with_full_model:
        env = HumanoidNumpyEnv('./humanoid_bench/assets/mjx/h1_pos_walk.xml', task=args.task)
    else:
        env = HumanoidNumpyEnv('./humanoid_bench/assets/mjx/h1_pos_walk.xml', task=args.task)
    
    state = env.reset()
    print("State:", state)

    if args.task == 'reach':
        torch_model = TorchModel(55, 19)
    elif args.task == 'reach_two_hands':
        torch_model = TorchModel(61, 19)
    elif args.task == 'walk' or args.task == 'stand':
        torch_model = TorchModel(51, 19)
    torch_policy = TorchPolicy(torch_model)

    torch_policy.load(args.model_file,
                    mean=args.model_file.replace('torch_model', 'mean').replace('.pt', '.npy'), 
                    var=args.model_file.replace('torch_model', 'var').replace('.pt', '.npy'))
    
    m, d = env.model, env.data

    if args.render:
        rollout_number = 1
        all_rewards = []
        all_videos = []

        renderer = mujoco.Renderer(m, height=480, width=480)
        for _ in tqdm.tqdm(range(rollout_number)):
            state = env.reset()
            i = 0
            reward = 0
            video = []
            while True:
                action = torch_policy(state)
                state, r, done, _ = env.step(action)
                reward += r
                i += 1
                renderer.update_scene(d, camera='cam_default')
                frame = renderer.render()
                video.append(frame)
                if done or i > 1000:
                    break
            all_rewards.append(reward)
            all_videos.append(np.array(video))
        make_grid_video_from_numpy(all_videos, 1, 
                                   output_name=args.model_file.replace('torch_model', 'evaluation').replace('.pt', '.mp4'), 
                                   **{'fps': 50})
        print("Rewards:", all_rewards)
    else:
        renderer = mujoco.Renderer(m, height=480, width=480)
        def get_image():
            state = env.reset()
            while True:
                action = torch_policy(state)
                state, _, _, _ = env.step(action)
                renderer.update_scene(d, camera='cam_default')
                img = renderer.render()
                
                # time.sleep(0.02)
                yield img

        fig = plt.figure(figsize=(6, 6))
        ani = animation.FuncAnimation(fig, update, frames=get_image(), interval=20)
        plt.show()
        


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default='./data/reach_one_hand/torch_model.pt')
    parser.add_argument('--task', type=str, default='reach')
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--with_full_model', action='store_true', default=False)
    args = parser.parse_args()

    main(args)
