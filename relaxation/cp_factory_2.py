import numpy as np
import matplotlib.pyplot as plt
import time
import jax
import jax.numpy as jnp 
jax.config.update("jax_enable_x64", True)
import modopt as mo
from typing import List, Callable
import gc
from combo import consensus


n = 30
dt = 2 / n
mc = 2
g = 9.81
d = 0.8
mu_cart = 0.03
mu_pole = 0.03
uscale = 10

initial_state_1 = np.array([0, np.pi, 0, 0])
initial_state_2 = np.array([1, np.pi, 0, 0])
initial_state_3 = np.array([0.5, np.pi, 0, 0])
initial_state_4 = np.array([0.75, np.pi, 0, 0])
# etc.
initial_states = [initial_state_1, initial_state_2]

N = len(initial_states) # number of blocks


objective = []
data = []


def make_functions(i):

    def block_i_solve(v_init: list,
                      y: np.ndarray = None, # lagrange multipliers
                      mu: float = 1, # penalty coefficient
                      ) -> list:
        
        print(f"function {i}")
        
        l1 = v_init[0] # dbl check this order!!!!!!!!!
        l2 = v_init[1]
        mp1 = v_init[2]
        mp2 = v_init[3]
        x1 = v_init[4]
        x2 = v_init[5]
        u1 = v_init[6]
        u2 = v_init[7]

        l_list = [l1, l2] # need to expand for changing N
        mp_list = [mp1, mp2] # need to expand for changing N
        x_list = [x1, x2] # need to expand for changing N
        u_list = [u1, u2] # need to expand for changing N

        u1 = u1 * uscale # ?????
        u2 = u2 * uscale # ?????

        j1 = 0.5 * dt * jnp.sum(u1[:-1]**2 + u1[1:]**2)
        j2 = 0.5 * dt * jnp.sum(u2[:-1]**2 + u2[1:]**2)
        j_list = [j1, j2] # need to expand for changing N

        def jax_obj(v):
            li = v[0]
            mpi = v[1]

            l_list[i] = li
            mp_list[i] = mpi

            c_l = consensus(l_list) # consensus for l
            c_mp = consensus(mp_list) # consensus for mp
            c = jnp.concatenate((c_l, c_mp)) # consensus for all global vars
            

            ui = v[n*4+2:].reshape((1, n)) * uscale

            ji = 0.5 * dt * jnp.sum(ui[0, :-1]**2 + ui[0, 1:]**2)

            j = 0
            for k in range(N):
                if k == i:
                    j += ji
                else:
                    j += j_list[k]

            return 1e-2 * j + y.T @ c + mu * jnp.sum(c**2)
        
        def jax_con(v):
            l = v[0]
            l_hat = l / 2
            mp = v[1]
            x = v[2:n*4+2].reshape((4, n))
            u = v[n*4+2:].reshape((n)) * uscale
            theta = x[1, :]
            dx = x[2, :]
            dtheta = x[3, :]
            cart_force = u
            sin_theta = jnp.sin(theta)
            cos_theta = jnp.cos(theta)
            ddx_num = (
                mp * g * sin_theta * cos_theta
                - (7 / 3) * (cart_force + mp * l_hat * dtheta**2 * sin_theta - mu_cart * dx)
                - (mu_pole * dtheta * cos_theta / l_hat))
            ddx_den = mp * cos_theta**2 - (7 / 3) * (mc + mp)
            ddx = ddx_num / ddx_den
            ddtheta = 3 * (g * sin_theta - ddx * cos_theta - (mu_pole * dtheta / (mp * l_hat))) / (7 * l_hat)
            f = jnp.vstack((dx, dtheta, ddx, ddtheta))
            c = x[:, 1:] - x[:, :-1] - 0.5 * dt * (f[:, 1:] + f[:, :-1])
            return 10 * c.flatten()
        
        # control bounds
        vl_u = np.full((n), -50 / uscale)
        vu_u = np.full((n),  50 / uscale)
        # initial condition
        vl_x = np.full((4, n), -np.inf)
        vu_x = np.full((4, n),  np.inf)
        vl_x[:, 0] = initial_states[i]
        vu_x[:, 0] = initial_states[i]
        # final condition
        vl_x[:, -1] = np.array([d, 0, 0, 0])
        vu_x[:, -1] = np.array([d, 0, 0, 0])
        # concatenate all bounds
        vl = np.concatenate((np.array([0.1, 0.1]), vl_x.flatten(), vl_u))
        vu = np.concatenate((np.array([5.0, np.inf]), vu_x.flatten(), vu_u))
        nc = 4 * (n - 1)  # dynamics constraints
        # initial guesses
        x0 = np.concatenate((np.array([l_list[i], mp_list[i]]), x_list[i].flatten(), u_list[i])) # unsure about these lists................................

        jaxprob = mo.JaxProblem(x0=x0, nc=nc, jax_obj=jax_obj, jax_con=jax_con,
                                order=1, xl=vl, xu=vu, cl=0., cu=0.)

        optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 700, 'ftol': 1e-9}, turn_off_outputs=True)
        optimizer.solve()
        optimizer.print_results()
        ans = optimizer.results['x']
        obj = optimizer.results['fun']
        objective.append(obj)
        gc.collect()

        l_list[i] = ans[0]
        mp_list[i] = ans[1]
        x_list[i] = ans[2:n*4+2].reshape((4, n))
        u_list[i] = ans[n*4+2:].reshape((n,))

        return l_list + mp_list + x_list + u_list



    return block_i_solve



functions = []
for i in range(N): # make a function for each block
    functions.append(make_functions(i))





def constraint(x_init: List[np.ndarray]) -> jnp.ndarray:
    
    l1 = x_init[0] # need to expand for changing N
    l2 = x_init[1] 
    mp1 = x_init[2]
    mp2 = x_init[3]

    c_l = consensus([l1, l2]) # need to expand for changing N
    c_mp = consensus([mp1, mp2]) # need to expand for changing N
    return jnp.concatenate((c_l, c_mp))








class GSCOptALR():
    def __init__(self, 
                 blocks: List[Callable],
                 constraint: Callable,
                 x_init: List[np.ndarray]):
        
        self.blocks = blocks
        self.x_init = x_init
        self.num_vars = len(x_init)
        self.success = False
        self.solution = None
        self.num_iter = 0
        self.time = None
        self.constraint = constraint
        self.mu = 1.0 # augmented Lagrangian penalty coefficient
        self.y = np.zeros_like(constraint(x_init)) # Lagrange multipliers

    def solve(self, 
              max_iter: int=100, 
              tol: float=1e-5,
              rho: float=1.2, # penalty increase factor
              ctol: float=1e-4, # consensus constraint tolerance
              ) -> bool:
        
        # check if rho is greater than 1
        if rho <= 1: raise ValueError("rho must be greater than 1")


        t1 = time.time()

        for k in range(max_iter):

            x_k_minus_1 = self.x_init.copy()

            for block in self.blocks:
                self.x_init = block(self.x_init, self.y, self.mu)



            # evaluate the consensus constraint
            c = self.constraint(self.x_init)

            # Check convergence
            if all(np.allclose(new, old, rtol=tol) 
                   for new, old in zip(self.x_init, x_k_minus_1)) and all(np.abs(c) < ctol):
                self.success = True
                break

            # prevent overflow
            if any(np.abs(c)) > ctol:

                # Update the Lagrange multipliers
                self.y = self.y + self.mu * c
                data.append(self.y)
                
                # Update the penalty coefficient
                self.mu = rho * self.mu


        self.num_iter = k + 1
        self.time = time.time() - t1
        self.solution = self.x_init

        return self.success






q1_0 = np.linspace(0, d, n)
q2_0 = np.linspace(np.pi, 0, n)
q3_0 = np.zeros(n)
q4_0 = np.zeros(n)
x0 = np.vstack((q1_0, q2_0, q3_0, q4_0))
u0 = np.zeros((n))
l0 = 0.5
mp0 = 0.4

# v_init = [l0, l0, mp0, mp0, x0, x0, u0, u0]
v_init = []
# automate the construction of v_init for changing N
for i in range(N): v_init.append(l0)
for i in range(N): v_init.append(mp0)
for i in range(N): v_init.append(x0)
for i in range(N): v_init.append(u0)

opt = GSCOptALR(blocks=functions, 
                constraint=constraint,
                x_init=v_init)

opt.solve(max_iter=100, 
          rho=1.2, # must be greater than 1
          tol=1e-4,
          ctol=1e-4)


print('Success: ', opt.success)
print('Iterations: ', opt.num_iter)
print('Time (s): ', opt.time)
print('Objective: ', objective[-1])

l1, mp1, l2, mp2, x1, u1, x2, u2 = opt.solution
print('l1: ', l1)
print('mp1: ', mp1)
print('l2: ', l2)
print('mp2: ', mp2)

plt.plot(objective[1:]) # skip the first infeasible iteration
plt.xlabel('Iteration')
plt.ylabel('Objective function value')
plt.grid()
plt.show()

plt.plot(data)
plt.xlabel('Iteration')
plt.grid()
plt.show()