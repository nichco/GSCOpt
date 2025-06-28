import numpy as np
import jax
import jax.numpy as jnp 
jax.config.update("jax_enable_x64", True)
import modopt as mo
import gc
import time
import tracemalloc

# generic problem definition using functions

def block1_solve(x_init: list, y=None):
    # design variable(s): x1

    x1_init = x_init[0]
    x2_init = x_init[1]

    def jax_obj(x):
        x1 = x[0]
        x2 = x2_init[0]
        return (1 - x1)**2 + 1*(x2 - x1**2)**2


    prob = mo.JaxProblem(x0=x1_init, jax_obj=jax_obj, xl=np.array([-np.inf]), xu=np.array([np.inf]), order=1)

    optimizer = mo.SLSQP(prob, solver_options={'maxiter': 20}, turn_off_outputs=True)
    optimizer.solve()
    optimizer.print_results()

    x1_star = optimizer.results['x']

    return [x1_star, x2_init]  # Return the updated x1 and the unchanged x2


def block2_solve(x_init: list, y=None):
    # design variable(s): x2

    x1_init = x_init[0]
    x2_init = x_init[1]

    def jax_obj(x):
        x1 = x1_init[0]
        x2 = x[0]
        return (1 - x1)**2 + 1*(x2 - x1**2)**2


    prob = mo.JaxProblem(x0=x2_init, jax_obj=jax_obj, xl=np.array([-np.inf]), xu=np.array([np.inf]), order=1)

    optimizer = mo.SLSQP(prob, solver_options={'maxiter': 20}, turn_off_outputs=True)
    optimizer.solve()
    optimizer.print_results()

    x2_star = optimizer.results['x']

    return [x1_init, x2_star]  # Return the unchanged x1 and the updated x2




class GSCOpt():
    def __init__(self, blocks, x_init: list):
        self.blocks = blocks
        self.x_init = x_init
        self.num_vars = len(x_init)
        self.success = False
        self.iter = 0

    def solve(self, max_iter=100, tol=1e-5):

        for k in range(max_iter):

            old= np.concatenate([x.flatten() for x in self.x_init])

            # Solve each block
            for block in self.blocks:
                solution = block(self.x_init)
                print('solution: ', solution)
                self.x_init = solution

                gc.collect()  # Clear memory after each iteration

            # Check convergence
            new = np.concatenate([x.flatten() for x in self.x_init])
            if np.allclose(new, old, rtol=tol):
                self.success = True
                break

        
        self.iter = k

        return




x_init = [np.array([5.]), np.array([5.])]
# print(np.array(x_init).shape)
# print(x_init[0].shape, x_init[1].shape)
# print('test: ', x_init[0].dtype)

# block coordinate descent algorithm
opt = GSCOpt(blocks=[block1_solve, block2_solve], x_init=x_init)


tracemalloc.start()  # Start tracing memory allocations

opt.solve(max_iter=50, tol=1e-4)
print('Success: ', opt.success)
print('Iterations: ', opt.iter)

# print peak memory usage
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()  # Stop tracing memory allocations
print(f"Peak memory usage: {peak / 10**6} MB")




# augmented Lagrangian modification