import jax
import jax.numpy as jp
from jax.random import PRNGKey
from humanoid_bench.mjx.envs.reach_continual import HumanoidReachContinual
from humanoid_bench.mjx.envs.walk import HumanoidWalkPosControl

def test_walk_task():
    # Initialize random key
    rng = PRNGKey(0)


    # Initialize the environment
    # env = HumanoidReachContinual(collisions='feet', 
    #                              act_control='pos', 
    #                              hands='both',
    #                              reward_weights_dict= {
    #                                 'alive': 1.0,            # Rewards for staying alive
    #                                 'vel': 0.5,              # Emphasize forward velocity for walking behavior
    #                                 'orientation': 0.2,      # Minor reward for maintaining orientation
    #                                 'walk_dist': 1.0,        # Added reward weight to encourage forward movement
    #                                 'efficiency': 0.5        # Penalize unnecessary actions to promote efficient movement
    #                             })
    env = HumanoidWalkPosControl(collisions='feet', act_control='pos', 
                                 reward_weights_dict={})
    
    # Reset the environment to get initial state
    state = env.reset(rng)
    
    # Print initial observation
    print("Initial Observation:", state.obs.shape)

    # Run a sample episode
    for step in range(10):  # Test with a few steps
        action = jp.ones(env.low_action.shape) * 0.5 # Zero action for testing
        print ("Action: ", action.shape)
        state = env.step(state, action)
        
        # Print the results of each step
        print(f"Step {step+1}")
        print("Observation:", state.obs)
        print("Reward:", state.reward)
        print("Terminated:", state.done)
        
        if state.done:
            print("Terminated early!")
            break

if __name__ == "__main__":
    test_walk_task()
