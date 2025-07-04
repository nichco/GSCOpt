from itertools import combinations
import jax.numpy as jnp




def consensus(variables: list) -> jnp.ndarray:

    # # ensure that every variable is the same shape
    # if not all(var.shape == variables[0].shape for var in variables):
    #     raise ValueError("consensus variables must have the same shape.")


    num_blocks = len(variables)
    indices = list(range(num_blocks))
    pairs = list(combinations(indices, 2))

    c = jnp.array([variables[i] - variables[j] for i, j in pairs])

    return c.flatten()




if __name__ == "__main__":
    # Example usage
    # Define some example variables
    l1 = jnp.array([1.0])
    l2 = jnp.array([2.0])
    l3 = jnp.array([3.0])

    # Call the consensus function
    c = consensus([l1, l2, l3])
    print(c)