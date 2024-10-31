import jax
from jax import numpy as jp
import mujoco
from brax.envs.base import Env, MjxEnv, State
from humanoid_bench.mjx.envs.utils import perturbed_pipeline_step

_STAND_HEIGHT = 1.65
_WALK_SPEED = 1

class HumanoidWalkPosControl(MjxEnv):
    def __init__(self, path="./unitree_h1/scene.xml", **kwargs):

        collisions = kwargs.get('collisions', 'feet')
        act_control = kwargs.get('act_control', 'pos')
        hands = kwargs.get('hands', 'both')

        path = "./humanoid_bench/assets/mjx/h1_pos_walk.xml"

        del kwargs['collisions']
        del kwargs['act_control']
        del kwargs['hands']

        super().__init__(model=mujoco.MjModel.from_xml_path(path), **kwargs)
        self.q_pos_init = jp.array(
            [0, 0, 0.98, 1, 0, 0, 0, 0, 0, -0.4, 0.8, -0.4, 0, 0, -0.4, 0.8, -0.4, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        )
        self.q_vel_init = jp.zeros(self.sys.nv)
        self.body_idxs = jp.array([i for i in range(self.sys.njnt)])
        self._move_speed = _WALK_SPEED
        self.low_action = jp.array(self.sys.actuator_ctrlrange[:, 0])
        self.high_action = jp.array(self.sys.actuator_ctrlrange[:, 1])

    def tolerance(self, x, bounds=(0.0, 1.0), margin=0.0, value_at_margin=0.0, sigmoid="linear"):
        lower, upper = bounds
        scale = jp.inf if margin == 0 else 1 / margin if sigmoid == "linear" else jp.log(1 / (1 - value_at_margin) - 1) / margin
        if sigmoid == "linear":
            below = jp.maximum(1.0 - scale * (lower - x), 0.0)
            above = jp.maximum(1.0 - scale * (x - upper), 0.0)
        elif sigmoid == "logistic":
            below = 1 / (1 + jp.exp(scale * (lower - x)))
            above = 1 / (1 + jp.exp(scale * (x - upper)))
        elif sigmoid == "quadratic":
            below = jp.maximum(1.0 - scale**2 * (lower - x)**2, 0.0)
            above = jp.maximum(1.0 - scale**2 * (x - upper)**2, 0.0)
        else:
            raise ValueError(f"Unsupported sigmoid type: {sigmoid}")
        return below * above

    def reset(self, rng):
        qpos = self.q_pos_init.copy()
        qvel = self.q_vel_init.copy()
        data = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(data.data)
        state = State(data, obs, jp.zeros(()), jp.zeros(()), {}, {'rng': rng, 'step_counter': 0})
        return state

    def _get_obs(self, data) -> jp.ndarray:
        return jp.concatenate([data.qpos[self.body_idxs], data.qvel[self.body_idxs]])

    def get_terminated(self, state):
        return state.pipeline_state.data.qpos[2] < 0.2

    def compute_reward(self, data, info):
        head_height = data.data.xpos[1, 2]  # Assuming this matches self.robot.head_height()
        com_velocity = data.data.qvel[0]    # Assuming this matches the x-component for forward speed
        
        # Standing reward calculation
        standing = self.tolerance(
            head_height, bounds=(_STAND_HEIGHT, float("inf")), margin=_STAND_HEIGHT / 4
        )
        upright = self.tolerance(
            jp.array(data.data.xmat[1, -1]),
            bounds=(0.9, float("inf")), margin=1.9, value_at_margin=0, sigmoid="linear"
        )
        stand_reward = standing * upright

        # Control penalty
        small_control = self.tolerance(
            data.data.qfrc_actuator, margin=10, value_at_margin=0, sigmoid="quadratic"
        ).mean()
        small_control = (4 + small_control) / 5

        if self._move_speed == 0:
            horizontal_velocity = data.data.qvel[:2]
            dont_move = self.tolerance(horizontal_velocity, margin=2).mean()
            reward = small_control * stand_reward * dont_move
            return reward, {
                "stand_reward": stand_reward,
                "small_control": small_control,
                "dont_move": dont_move,
                "standing": standing,
                "upright": upright,
            }
        else:
            move = self.tolerance(
                com_velocity, bounds=(self._move_speed, float("inf")),
                margin=self._move_speed, value_at_margin=0, sigmoid="linear"
            )
            move = (5 * move + 1) / 6
            reward = small_control * stand_reward * move
            return reward, {
                "stand_reward": stand_reward,
                "small_control": small_control,
                "move": move,
                "standing": standing,
                "upright": upright,
            }


    def unnorm_action(self, action):
        return (action + 1) / 2 * (self.high_action - self.low_action) + self.low_action
        
    def step(self, state: State, action: jp.ndarray) -> State:
        action = self.unnorm_action(action)
        data = perturbed_pipeline_step(self.sys, state.pipeline_state, action, jp.zeros((self.sys.nbody, 6)), self._n_frames)
        terminated = self.get_terminated(state)
        reward, info = self.compute_reward(data, state.info)
        obs = self._get_obs(data.data)
        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=terminated, info=info)
