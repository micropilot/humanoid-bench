import jax
import jax.numpy as jp
from jax.random import PRNGKey
from humanoid_bench.mjx.envs.walk import HumanoidWalkPosControl

def test_walk_task():
    # Initialize random key
    rng = PRNGKey(0)


    # Initialize the environment
    env = HumanoidWalkPosControl()
    
    # Reset the environment to get initial state
    state = env.reset(rng)
    
    # Print initial observation
    print("Initial Observation:", state.obs.shape)

    # Run a sample episode
    for step in range(10):  # Test with a few steps
        action = jp.ones(env.low_action.shape) * 0.5 # Zero action for testing
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
