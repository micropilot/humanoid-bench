import jax
from jax import numpy as jp
import mujoco
from brax.envs.base import Env, MjxEnv, State
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

    def tolerance(self, x, bounds=(0.0, 1.0), margin=0.0, value_at_margin=0.0, sigmoid="linear"):
        lower, upper = bounds
        
        if margin == 0:
            scale = 1e10  # Large finite value for zero margin
        else:
            if sigmoid == "linear":
                scale = 1 / margin
            elif sigmoid == "logistic":
                # Avoid issues with extreme value_at_margin inputs by clamping
                value_at_margin = jp.clip(value_at_margin, 1e-5, 1 - 1e-5)
                scale = jp.log(1 / (1 - value_at_margin) - 1) / margin
            elif sigmoid == "quadratic":
                scale = 1 / (margin ** 2)
            else:
                raise ValueError(f"Unsupported sigmoid type: {sigmoid}")
            
        if sigmoid == "linear":
            below = jp.clip(1.0 - scale * (lower - x), 0.0, 1.0)
            above = jp.clip(1.0 - scale * (x - upper), 0.0, 1.0)
        elif sigmoid == "logistic":
            below = 1 / (1 + jp.exp(scale * (lower - x)))
            above = 1 / (1 + jp.exp(scale * (x - upper)))
        elif sigmoid == "quadratic":
            below = jp.clip(1.0 - scale * (lower - x)**2, 0.0, 1.0)
            above = jp.clip(1.0 - scale * (x - upper)**2, 0.0, 1.0)

        return below * above

    def reset(self, rng):
        step_counter = 0 

        qpos = self.q_pos_init.copy()
        qvel = self.q_vel_init.copy()
        
        data = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(data.data)
        state = State(data, obs, jp.zeros(()), jp.zeros(()), {}, 
                      {'rng': rng, 
                       'step_counter': 0,
                       'last_xfrc_applied': jp.zeros((self.sys.nbody, 6)),
                       'success': 0.,
                       'total_successes': 0.})
        return state

    def _get_obs(self, data) -> jp.ndarray:
        return jp.concatenate([data.qpos[self.body_idxs][2:],
                data.qvel[self.body_vel_idxs]])

    def get_terminated(self, state):
        return state.pipeline_state.data.qpos[2] < 0.2

    def compute_reward(self, data, info):
        head_height = data.data.site_xpos[5, -1]
        # Standing reward calculation
        standing = self.tolerance(
            head_height, bounds=(_STAND_HEIGHT, float("inf")), margin=_STAND_HEIGHT / 4
        )
        if jp.isnan(standing).any():
            print("NaN detected in standing tolerance calculation.")
        
        torso_upright = data.data.xmat[1, -1]
        upright = self.tolerance(
            torso_upright,
            bounds=(0.9, float("inf")), margin=1.9, value_at_margin=0, sigmoid="linear"
        )
        if jp.isnan(upright).any():
            print("NaN detected in upright tolerance calculation.")

        stand_reward = standing * upright

        # Control penalty
        actuator_force = data.data.qfrc_actuator
        small_control = self.tolerance(
            actuator_force, margin=10, value_at_margin=0, sigmoid="quadratic"
        ).mean()
        small_control = (4 + small_control) / 5
        if jp.isnan(small_control).any():
            print("NaN detected in small_control calculation.")

        # Condition for move or no move
        print (data)
        if self._move_speed == 0:
            horizontal_velocity = data.data.sensordata[2:4]
            dont_move = self.tolerance(horizontal_velocity, margin=2).mean()
            if jp.isnan(dont_move).any():
                print("NaN detected in dont_move calculation.")
            
            reward = small_control * stand_reward * dont_move
            return reward, {
                "stand_reward": stand_reward,
                "small_control": small_control,
                "dont_move": dont_move,
                "standing": standing,
                "upright": upright,
            }
        else:
            com_velocity = data.data.sensordata[2]
            move = self.tolerance(
                com_velocity, bounds=(self._move_speed, float("inf")),
                margin=self._move_speed, value_at_margin=0, sigmoid="linear"
            )
            move = (5 * move + 1) / 6
            if jp.isnan(move).any():
                print("NaN detected in move calculation.")
            
            reward = small_control * stand_reward * move
            return reward, {
                "stand_reward": stand_reward,
                "small_control": small_control,
                "move": move,
                "standing": standing,
                "upright": upright,
            }

    def get_info(self, state, data):
        """
        Computes relevant metrics for the HumanoidWalkPosControl environment.

        Args:
            state: The current state of the environment.
            data: The environment data from MuJoCo containing position, velocity, and other dynamic information.

        Returns:
            A dictionary of computed metrics related to the humanoid's walking stability, speed, and control.
        """
        # Get torso and head positions
        torso_pos = data.data.xpos[1]
        head_height = torso_pos[2]

        # Center of mass velocity for forward movement
        com_velocity = data.data.qvel[0]  # Forward velocity on x-axis
        
        # Stand upright status (based on torso orientation matrix)
        uprightness = jp.array(data.data.xmat[1, -1])  # Orientation to measure uprightness

        # Maximum joint velocity for control stability
        max_joint_vel = jp.max(jp.abs(data.data.qvel[self.body_vel_idxs]))

        # Define metrics for rewards
        standing = self.tolerance(head_height, bounds=(_STAND_HEIGHT, float("inf")), margin=_STAND_HEIGHT / 4)
        upright = self.tolerance(uprightness, bounds=(0.9, float("inf")), margin=1.9, value_at_margin=0, sigmoid="linear")

        # Accumulate success if the humanoid maintains an upright, stable walk
        is_standing = jp.where(standing * upright > 0.8, 1.0, 0.0)  # Threshold to consider "standing"
        total_successes = state.info['total_successes'] + is_standing

        return {
            'head_height': head_height,
            'com_velocity': com_velocity,
            'uprightness': uprightness,
            'max_joint_vel': max_joint_vel,
            'standing': standing,
            'upright': upright,
            'is_standing': is_standing,
            'total_successes': total_successes
        }


    def unnorm_action(self, action):
        return (action + 1) / 2 * (self.high_action - self.low_action) + self.low_action
        
    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""

        # Manage perturbations and actions
        rng, subkey = jax.random.split(state.info['rng'])
        action = self.unnorm_action(action)

        apply_every, hold_for, magnitude = 1, 1, 1
        xfrc_applied = jax.lax.cond(
            state.info['step_counter'] % apply_every == 0,
            lambda _: jax.random.normal(subkey, shape=(self.sys.nbody, 6)) * magnitude,
            lambda _: state.info['last_xfrc_applied'], operand=None
        )
        perturb = jax.lax.cond(
            state.info['step_counter'] % apply_every < hold_for, lambda _: 1, lambda _: 0, operand=None
        )
        xfrc_applied = xfrc_applied * perturb

        # Run dynamics with perturbed step
        data = perturbed_pipeline_step(self.sys, state.pipeline_state, action, xfrc_applied, self._n_frames)
        observation = self._get_obs(data.data)

        # Call `get_info` to gather info without direct modification
        log_info = self.get_info(state, data)

        # Compute reward based on new data
        reward, terminated = self.compute_reward(data, log_info)

        # Update `state.info` consistently
        state.info.update(
            rng=rng,
            step_counter=state.info['step_counter'] + 1,
            last_xfrc_applied=xfrc_applied,
        )
        state.info.update(**log_info)

        return state.replace(
            pipeline_state=data,
            obs=observation,
            reward=reward,
            done=terminated
     )
