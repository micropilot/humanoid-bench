import jax
from jax import lax
from jax import numpy as jp
import mujoco
from brax.envs.base import Env, MjxEnv, State
from .rewards import tolerance

from humanoid_bench.mjx.envs.utils import perturbed_pipeline_step

_STAND_HEIGHT = 1.65
_WALK_SPEED = 1

class HumanoidWalkPosControl(MjxEnv):
    def __init__(self, path="./unitree_h1/scene.xml", reward_weights_dict=None, **kwargs):

        collisions = kwargs.get('collisions', 'feet')
        act_control = kwargs.get('act_control', 'pos')

        path = "./humanoid_bench/assets/mjx/h1_pos_walk.xml"

        del kwargs['collisions']
        del kwargs['act_control']

        mj_model = mujoco.MjModel.from_xml_path(path)

        physics_steps_per_control_step = 10
        kwargs['n_frames'] = kwargs.get(
            'n_frames', physics_steps_per_control_step)
        
        self.body_idxs = []
        self.body_vel_idxs = []
        curr_idx = 0
        curr_vel_idx = 0
        for i in range(mj_model.njnt):
            joint_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name.startswith('free_'):
                if joint_name == 'free_base':
                    self.body_idxs.extend(list(range(curr_idx, curr_idx + 7)))
                    self.body_vel_idxs.extend(list(range(curr_vel_idx, curr_vel_idx + 6)))
                curr_idx += 7
                curr_vel_idx += 6
                continue
            elif not joint_name.startswith('lh_') and not joint_name.startswith('rh_') and not 'wrist' in joint_name: # NOTE: excluding hands here
                self.body_idxs.append(curr_idx)
                self.body_vel_idxs.append(curr_vel_idx)
            curr_idx += 1
            curr_vel_idx += 1

        print("Body idxs: ", self.body_idxs)
        print("Body vel idxs: ", self.body_vel_idxs)

        self.body_idxs = jp.array(self.body_idxs)
        self.body_vel_idxs = jp.array(self.body_vel_idxs)

        super().__init__(model=mj_model, **kwargs)

        self.q_pos_init = jp.array(
            [0, 0, 0.98, 1, 0, 0, 0, 0, 0, -0.4, 0.8, -0.4, 0, 0, -0.4, 0.8, -0.4, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        )
        self.q_vel_init = jp.zeros(self.sys.nv)

        self._move_speed = _WALK_SPEED
        self._stand_height = _STAND_HEIGHT
        action_range = self.sys.actuator_ctrlrange
        self.low_action = jp.array(action_range[:, 0])
        self.high_action = jp.array(action_range[:, 1])

        data = self.pipeline_init(
            self.q_pos_init,
            self.q_vel_init,
        )

        self.state_dim = self._get_obs(data.data).shape[-1]
        self.action_dim = self.sys.nu

        assert reward_weights_dict is not None
        self.reward_weight_dict = reward_weights_dict

    def reset(self, rng):
        step_counter = 0 

        qpos = self.q_pos_init.copy()
        qvel = self.q_vel_init.copy()
        
        reward, done, zero = jp.zeros(3)
        data = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(data.data)
        state = State(data, obs, reward, done, 
                      {'reward': zero},
                      {'rng': rng, 
                       'step_counter': step_counter,
                       'last_xfrc_applied': jp.zeros((self.sys.nbody, 6)),
                       "stand_reward": 0., 
                       "small_control": 0., 
                       "move": 0.,
                       "standing": 0.,
                       "upright": 0.,
                       })
        
        return state

    def compute_reward(self, data):
        # Standing and upright calculations
        head_height = data.data.site_xpos[2, -1]
        standing = tolerance(head_height, bounds=(self._stand_height, float("inf")), margin=0.4125)
        
        torso_upright = data.data.xmat[1, 2, 2]
        upright = tolerance(torso_upright, bounds=(0.9, float("inf")), sigmoid="linear", margin=1.9, value_at_margin=0)
        stand_reward = standing * upright

        # Small control penalty
        actuator_force = data.data.qfrc_actuator
        small_control = tolerance(actuator_force, margin=10, value_at_margin=0, sigmoid="quadratic").mean()
        small_control = (4 + small_control) / 5

        # Movement calculation
        com_velocity = data.data.qvel[0]
        move = tolerance(com_velocity, bounds=(self._move_speed, float("inf")), margin=self._move_speed, value_at_margin=0, sigmoid="linear")
        move = (5 * move + 1) / 6

        # Basic reward with movement
        reward = small_control * stand_reward * move

        # Termination condition
        terminated = jp.where(data.data.qpos[2] < 0.2, 1.0, 0.0)
        reward = jp.where(jp.isnan(reward), 0, reward)

        sub_rewards = {
            "stand_reward": stand_reward, 
            "small_control": small_control, 
            "move": move,
            "standing": standing,
            "upright": upright,
        }

        return reward, terminated, sub_rewards

    def unnorm_action(self, action):
        return (action + 1) / 2 * (self.high_action - self.low_action) + self.low_action
    
    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""

        apply_every = 1
        hold_for = 1
        magnitude = 1

        # Reset the applied forces every 200 steps
        rng, subkey = jax.random.split(state.info['rng'])
        xfrc_applied = jp.zeros((self.sys.nbody, 6))
        xfrc_applied = jax.lax.cond(
            state.info['step_counter'] % apply_every == 0,
            lambda _: jax.random.normal(subkey, shape=(self.sys.nbody, 6)) * magnitude,
            lambda _: state.info['last_xfrc_applied'], operand=None)
        # Reset to 0 every 50 steps
        perturb = jax.lax.cond(
            state.info['step_counter'] % apply_every < hold_for, lambda _: 1, lambda _: 0, operand=None)
        xfrc_applied = xfrc_applied * perturb

        action = self.unnorm_action(action)

        # Run dynamics with perturbed step
        data = perturbed_pipeline_step(self.sys, state.pipeline_state, action, xfrc_applied, self._n_frames)
        observation = self._get_obs(data.data)

        # Compute reward based on new data
        reward, terminated, sub_rewards = self.compute_reward(data)

        # Update `state.info` consistently
        state.info.update(
            rng=rng,
            step_counter=state.info['step_counter'] + 1,
            last_xfrc_applied=xfrc_applied,
        )
        state.info.update(**sub_rewards)

        return state.replace(
            pipeline_state=data,
            obs=observation,
            reward=reward,
            done=terminated
     )
    
    def _get_obs(self, data) -> jp.ndarray:
        return jp.concatenate([data.qpos[self.body_idxs],
                data.qvel[self.body_vel_idxs]])

