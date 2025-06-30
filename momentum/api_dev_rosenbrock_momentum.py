import numpy as np
import jax
import jax.numpy as jnp 
jax.config.update("jax_enable_x64", True)
import modopt as mo
import gc
import time
import matplotlib.pyplot as plt
import pickle


objective = []

# generic problem definition using functions

def block1_solve(x_init: list, y=None):
    # design variable(s): x1

    x1_init = x_init[0]
    x2_init = x_init[1]

    def jax_obj(x):
        x1 = x[0]
        x2 = x2_init[0]
        return (1 - x1)**2 + 10*(x2 - x1**2)**2


    prob = mo.JaxProblem(x0=x1_init, 
                         jax_obj=jax_obj, 
                         xl=np.array([-np.inf]), 
                         xu=np.array([np.inf]), 
                         order=1)

    optimizer = mo.SLSQP(prob, solver_options={'maxiter': 20}, turn_off_outputs=True)
    optimizer.solve()
    optimizer.print_results()

    x1_star = optimizer.results['x']
    obj = optimizer.results['fun']
    objective.append(obj)

    return [x1_star, x2_init]  # Return the updated x1 and the unchanged x2


def block2_solve(x_init: list, y=None):
    # design variable(s): x2

    x1_init = x_init[0]
    x2_init = x_init[1]

    def jax_obj(x):
        x1 = x1_init[0]
        x2 = x[0]
        return (1 - x1)**2 + 10*(x2 - x1**2)**2


    prob = mo.JaxProblem(x0=x2_init, 
                         jax_obj=jax_obj, 
                         xl=np.array([-np.inf]), 
                         xu=np.array([np.inf]), 
                         order=1)

    optimizer = mo.SLSQP(prob, solver_options={'maxiter': 20}, turn_off_outputs=True)
    optimizer.solve()
    optimizer.print_results()

    x2_star = optimizer.results['x']
    obj = optimizer.results['fun']
    objective.append(obj)

    return [x1_init, x2_star]  # Return the unchanged x1 and the updated x2




class GSCOpt():
    def __init__(self, blocks, x_init: list):
        self.blocks = blocks
        self.x_init = x_init
        self.num_vars = len(x_init)
        self.success = False
        self.iter = 0
        self.time = None
        self.omega = 1.0

    def solve(self, 
              max_iter: int=100, 
              tol: float=1e-5):

        t1 = time.time()

        for k in range(max_iter):

            old = np.concatenate([x.flatten() for x in self.x_init])
            old_values = self.x_init.copy()

            for block in self.blocks:
                self.x_init = block(self.x_init)
                print('solution: ', self.x_init)

                gc.collect()  # Clear memory after each iteration

            # Check convergence
            new = np.concatenate([x.flatten() for x in self.x_init])
            if np.allclose(new, old, rtol=tol):
                self.success = True
                break

            # momentum update
            new_values = self.x_init.copy()
            for i in range(self.num_vars):
                self.x_init[i] = new_values[i] + self.omega * (new_values[i] - old_values[i])

            # omega update
            if k > 0:
                if objective[-1] < objective[-2]:
                    # self.omega *= 1.1
                    pass
                else:
                    self.omega *= 0.7

        self.iter = k
        self.time = time.time() - t1

        return





x_init = [np.array([5.]), np.array([5.])]

# block coordinate descent algorithm
opt = GSCOpt(blocks=[block1_solve, block2_solve], x_init=x_init)
opt.solve(max_iter=500, tol=1e-4)


print('Success: ', opt.success)
print('Iterations: ', opt.iter)
print('Time (s): ', opt.time)


# plt.plot(objective)
# plt.xlabel('Iteration')
# plt.ylabel('Objective function value')
# plt.grid()
# plt.show()





# with open('omega_0_b10.pkl', 'wb') as f:
#     pickle.dump(objective, f)



# open pickle file and plot
with open('momentum/omega_0_b10.pkl', 'rb') as f:
    objective_0 = pickle.load(f)

plt.plot(objective_0)
plt.plot(objective)
plt.xlabel('Iteration')
plt.ylabel('Objective function value')
plt.grid()
plt.show()