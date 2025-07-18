import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import jax
import jax.numpy as jnp 
jax.config.update("jax_enable_x64", True)
import modopt as mo
from typing import List, Callable
import gc
from combo import consensus
import tracemalloc


n = 30
dt = 2 / n
mc = 2
g = 9.81
d = 0.8
mu_cart = 0.03
mu_pole = 0.03
uscale = 10

initial_state_1 = np.array([0, np.pi, 0, 0])
initial_state_2 = np.array([0.5, np.pi, 0, 0])
initial_state_3 = np.array([0.25, np.pi, 0, 0])
initial_state_4 = np.array([0.75, np.pi, 0, 0])
initial_state_5 = np.array([-0.5, np.pi, 0, 0])
initial_state_6 = np.array([-0.25, np.pi, 0, 0])
initial_state_7 = np.array([0, np.pi+np.pi/2, 0, 0])
# etc.
initial_states = [initial_state_1, initial_state_2, initial_state_3, initial_state_4, initial_state_5, initial_state_6, initial_state_7]

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
        l3 = v_init[2]
        l4 = v_init[3]
        l5 = v_init[4]
        l6 = v_init[5]
        l7 = v_init[6]
        mp1 = v_init[7]
        mp2 = v_init[8]
        mp3 = v_init[9]
        mp4 = v_init[10]
        mp5 = v_init[11]
        mp6 = v_init[12]
        mp7 = v_init[13]
        x1 = v_init[14]
        x2 = v_init[15]
        x3 = v_init[16]
        x4 = v_init[17]
        x5 = v_init[18]
        x6 = v_init[19]
        x7 = v_init[20]
        u1 = v_init[21]
        u2 = v_init[22]
        u3 = v_init[23]
        u4 = v_init[24]
        u5 = v_init[25]
        u6 = v_init[26]
        u7 = v_init[27]

        l_list = [l1, l2, l3, l4, l5, l6, l7] # need to expand for changing N
        mp_list = [mp1, mp2, mp3, mp4, mp5, mp6, mp7] # need to expand for changing N
        x_list = [x1, x2, x3, x4, x5, x6, x7] # need to expand for changing N
        u_list = [u1, u2, u3, u4, u5, u6, u7] # need to expand for changing N

        u1 = u1 * uscale # ?????
        u2 = u2 * uscale # ?????
        u3 = u3 * uscale # ?????
        u4 = u4 * uscale # ?????
        u5 = u5 * uscale # ?????
        u6 = u6 * uscale # ?????
        u7 = u7 * uscale # ?????

        j1 = 0.5 * dt * np.sum(u1[:-1]**2 + u1[1:]**2)
        j2 = 0.5 * dt * np.sum(u2[:-1]**2 + u2[1:]**2)
        j3 = 0.5 * dt * np.sum(u3[:-1]**2 + u3[1:]**2)
        j4 = 0.5 * dt * np.sum(u4[:-1]**2 + u4[1:]**2)
        j5 = 0.5 * dt * np.sum(u5[:-1]**2 + u5[1:]**2)
        j6 = 0.5 * dt * np.sum(u6[:-1]**2 + u6[1:]**2)
        j7 = 0.5 * dt * np.sum(u7[:-1]**2 + u7[1:]**2)
        j_list = [j1, j2, j3, j4, j5, j6, j7] # need to expand for changing N

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
        vl_u = np.full((n), -40 / uscale)
        vu_u = np.full((n),  40 / uscale)
        # initial condition
        vl_x = np.full((4, n), -np.inf)
        vu_x = np.full((4, n),  np.inf)
        vl_x[1, :] = -1e-3 # lower bound on angle
        vl_x[0, :] = initial_states[i][0] - 1e-3  # lower bound on cart position
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

        optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 700, 'ftol': 1e-7}, turn_off_outputs=True)
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
    l3 = x_init[2]
    l4 = x_init[3]
    l5 = x_init[4]
    l6 = x_init[5]
    l7 = x_init[6]
    mp1 = x_init[7]
    mp2 = x_init[8]
    mp3 = x_init[9]
    mp4 = x_init[10]
    mp5 = x_init[11]
    mp6 = x_init[12]
    mp7 = x_init[13]

    c_l = consensus([l1, l2, l3, l4, l5, l6, l7]) # need to expand for changing N
    c_mp = consensus([mp1, mp2, mp3, mp4, mp5, mp6, mp7]) # need to expand for changing N
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

v_init = []
# automate the construction of v_init for changing N
for i in range(N): v_init.append(l0)
for i in range(N): v_init.append(mp0)
for i in range(N): v_init.append(x0)
for i in range(N): v_init.append(u0)


opt = GSCOptALR(blocks=functions, 
                constraint=constraint,
                x_init=v_init)

tracemalloc.start()

opt.solve(max_iter=100, 
          rho=1.2, # must be greater than 1
          tol=1e-7,
          ctol=1e-4)

# print peak memory usage
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"Peak memory usage: {peak / 1e6} MB")


print('Success: ', opt.success)
print('Iterations: ', opt.num_iter)
print('Time (s): ', opt.time)
print('Objective: ', objective[-1])

l_data = opt.solution[0*N : 1*N]
mp_data = opt.solution[1*N : 2*N]
x_data = opt.solution[2*N : 3*N]
u_data = opt.solution[3*N : 4*N]
print('l: ', l_data)
print('mp: ', mp_data)

plt.plot(objective[6:]) # skip the first infeasible iteration(s)
plt.xlabel('Iteration')
plt.ylabel('Objective function value')
plt.show()

plt.plot(data)
plt.xlabel('Iteration')
plt.ylabel('Lagrange multipliers')
plt.show()





import pickle
with open('obj7.pkl', 'wb') as f:
    pickle.dump(objective, f)




# plot the cart-pole trajectories
fig, axs = plt.subplots(1, 7, figsize=(21, 3))
axs = axs.flatten()

cart_width, cart_height = 0.2, 0.1
cmap   = cm.get_cmap('viridis', n)
colors = cmap(np.arange(n))
alpha = np.linspace(0.1, 1, n)

for i in range(N):
    x = x_data[i]
    l = l_data[i]

    position = x[0, :]  # cart position
    angle = x[1, :]    # pole angle
    pole_x = position + l * np.sin(angle)
    pole_y = l * np.cos(angle)

    ax = axs[i]

    for i in range(n):

        cart = plt.Rectangle((position[i] - cart_width/2, -cart_height/2), cart_width, cart_height, facecolor='lightgrey', edgecolor='black', alpha=alpha[i])
        ax.add_patch(cart)

        ax.plot([position[i], pole_x[i]], [0, pole_y[i]], linewidth=2, color='black', alpha=alpha[i])
        ax.scatter(pole_x[i], pole_y[i], color=colors[i], s=130, zorder=10, alpha=1, edgecolor='black') # mass at end of pole

        ax.set_facecolor('ghostwhite')

plt.show()