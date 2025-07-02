import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import jax
import jax.numpy as jnp 
jax.config.update("jax_enable_x64", True)
import modopt as mo
import gc

n = 30
dt = 2 / n
mc = 2
g = 9.81
d = 0.8
mu_cart = 0.03
mu_pole = 0.03
uscale = 10

rho = 10 # NEW PENALTY COEFFICIENT *******************

objective = []


def block1_solve(v_init: list, y=None):
    # design variable(s): all

    l_init_1 = v_init[0]
    mp_init_1 = v_init[1]
    l_init_2 = v_init[2]
    mp_init_2 = v_init[3]
    x_init_1 = v_init[4]
    u_init_1 = v_init[5]
    x_init_2 = v_init[6]
    u_init_2 = v_init[7]

    def jax_obj(v):
        l = v[0]
        mp = v[1]
        x0_1 = jnp.array([l, mp])
        x0_2 = jnp.array([l_init_2, mp_init_2])

        penalty = jnp.sum((x0_1 - x0_2)**2)

        u = v[n*4+2:].reshape((1, n)) * uscale
        j1 = 0.5 * dt * jnp.sum(u[0, :-1]**2 + u[0, 1:]**2)
        j2 = 0.5 * dt * jnp.sum(u_init_2[:-1]**2 + u_init_2[1:]**2)
        
        return 1e-2 * j1 + rho * penalty

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
            - (mu_pole * dtheta * cos_theta / l_hat)
        )

        ddx_den = mp * cos_theta**2 - (7 / 3) * (mc + mp)
        ddx = ddx_num / ddx_den

        ddtheta = 3 * (g * sin_theta - ddx * cos_theta - (mu_pole * dtheta / (mp * l_hat))) / (7 * l_hat)

        # Stack all into f: shape (4, n)
        f = jnp.vstack((dx, dtheta, ddx, ddtheta))

        c = x[:, 1:] - x[:, :-1] - 0.5 * dt * (f[:, 1:] + f[:, :-1])
        
        return 10 * c.flatten()
    

    # Compute the variable bounds

    vl_l = 0.1
    vu_l = 5.0
    vl_mp = 0.1
    vu_mp = np.inf

    # control bounds
    vl_u = np.full((n), -50 / uscale)
    vu_u = np.full((n),  50 / uscale)


    # initial condition
    vl_x = np.full((4, n), -np.inf)
    vu_x = np.full((4, n),  np.inf)

    vl_x[:, 0] = np.array([0, np.pi, 0, 0])
    vu_x[:, 0] = np.array([0, np.pi, 0, 0])

    # final condition
    vl_x[:, -1] = np.array([d, 0, 0, 0])
    vu_x[:, -1] = np.array([d, 0, 0, 0])

    # concatenate all bounds
    vl = np.concatenate((np.array([vl_l, vl_mp]), vl_x.flatten(), vl_u))
    vu = np.concatenate((np.array([vu_l, vu_mp]), vu_x.flatten(), vu_u))

    nc = 4 * (n - 1)  # dynamics constraints

    # initial guesses
    x0 = np.concatenate((np.array([l_init_1, mp_init_1]), x_init_1.flatten(), u_init_1))

    jaxprob = mo.JaxProblem(x0=x0, nc=nc, jax_obj=jax_obj, jax_con=jax_con,
                            order=1, xl=vl, xu=vu, cl=0., cu=0.)

    optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 500, 'ftol': 1e-9}, turn_off_outputs=True)
    optimizer.solve()
    optimizer.print_results()

    ans = optimizer.results['x']
    obj = optimizer.results['fun']
    objective.append(obj)

    return [ans[0], ans[1], l_init_2, mp_init_2, ans[2:n*4+2].reshape((4, n)), ans[n*4+2:].reshape((n,)), x_init_2, u_init_2]






def block2_solve(v_init: list, y=None):
    # design variable(s): all

    l_init_1 = v_init[0]
    mp_init_1 = v_init[1]
    l_init_2 = v_init[2]
    mp_init_2 = v_init[3]
    x_init_1 = v_init[4]
    u_init_1 = v_init[5]
    x_init_2 = v_init[6]
    u_init_2 = v_init[7]

    def jax_obj(v):
        l = v[0]
        mp = v[1]
        x0_1 = jnp.array([l_init_1, mp_init_1])
        x0_2 = jnp.array([l, mp])

        penalty = jnp.sum((x0_1 - x0_2)**2)

        u = v[n*4+2:].reshape((1, n)) * uscale
        j1 = 0.5 * dt * jnp.sum(u_init_1[:-1]**2 + u_init_1[1:]**2)
        j2 = 0.5 * dt * jnp.sum(u[0, :-1]**2 + u[0, 1:]**2)

        return 1e-2 * j2 + rho * penalty

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
            - (mu_pole * dtheta * cos_theta / l_hat)
        )

        ddx_den = mp * cos_theta**2 - (7 / 3) * (mc + mp)
        ddx = ddx_num / ddx_den

        ddtheta = 3 * (g * sin_theta - ddx * cos_theta - (mu_pole * dtheta / (mp * l_hat))) / (7 * l_hat)

        # Stack all into f: shape (4, n)
        f = jnp.vstack((dx, dtheta, ddx, ddtheta))

        c = x[:, 1:] - x[:, :-1] - 0.5 * dt * (f[:, 1:] + f[:, :-1])
        
        return 10 * c.flatten()
    

    # Compute the variable bounds

    vl_l = 0.1
    vu_l = 5.0
    vl_mp = 0.1
    vu_mp = np.inf

    # control bounds
    vl_u = np.full((n), -50 / uscale)
    vu_u = np.full((n),  50 / uscale)


    # initial condition
    vl_x = np.full((4, n), -np.inf)
    vu_x = np.full((4, n),  np.inf)

    vl_x[:, 0] = np.array([0, np.pi, 0, 0])
    vu_x[:, 0] = np.array([0, np.pi, 0, 0])

    # final condition
    vl_x[:, -1] = np.array([d, 0, 0, 0])
    vu_x[:, -1] = np.array([d, 0, 0, 0])

    # concatenate all bounds
    vl = np.concatenate((np.array([vl_l, vl_mp]), vl_x.flatten(), vl_u))
    vu = np.concatenate((np.array([vu_l, vu_mp]), vu_x.flatten(), vu_u))

    nc = 4 * (n - 1)  # dynamics constraints

    # initial guesses
    x0 = np.concatenate((np.array([l_init_2, mp_init_2]), x_init_2.flatten(), u_init_2))

    jaxprob = mo.JaxProblem(x0=x0, nc=nc, jax_obj=jax_obj, jax_con=jax_con,
                            order=1, xl=vl, xu=vu, cl=0., cu=0.)

    optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 500, 'ftol': 1e-9}, turn_off_outputs=True)
    optimizer.solve()
    optimizer.print_results()

    ans = optimizer.results['x']
    obj = optimizer.results['fun']
    objective.append(obj)

    return [l_init_1, mp_init_1, ans[0], ans[1], x_init_1, u_init_1, ans[2:n*4+2].reshape((4, n)), ans[n*4+2:].reshape((n,))]









class GSCOpt():
    def __init__(self, blocks, x_init: list):
        self.blocks = blocks
        self.x_init = x_init
        self.num_vars = len(x_init)
        self.success = False
        self.iter = 0
        self.time = None

    def solve(self, 
              max_iter: int=100, 
              tol: float=1e-5):

        t1 = time.time()

        for k in range(max_iter):

            old = np.concatenate([np.array(x).flatten() for x in self.x_init])

            for block in self.blocks:
                self.x_init = block(self.x_init)

                gc.collect()  # Clear memory after each iteration

            # Check convergence
            new = np.concatenate([np.array(x).flatten() for x in self.x_init])
            if np.allclose(new, old, rtol=tol):
                self.success = True
                break

        
        self.iter = k
        self.time = time.time() - t1

        return






q1_0 = np.linspace(0, d, n)
q2_0 = np.linspace(np.pi, 0, n)
q3_0 = np.zeros(n)
q4_0 = np.zeros(n)
x0 = np.vstack((q1_0, q2_0, q3_0, q4_0))
u0 = np.zeros((n))

l0 = 0.5
mp0 = 0.4

v_init = [l0, mp0, l0, mp0, x0, u0, x0, u0]

# block coordinate descent algorithm
opt = GSCOpt(blocks=[block1_solve, block2_solve], x_init=v_init)
opt.solve(max_iter=10, tol=1e-4)


print('Success: ', opt.success)
print('Iterations: ', opt.iter)
print('Time (s): ', opt.time)


plt.plot(objective)
plt.xlabel('Iteration')
plt.ylabel('Objective function value')
plt.grid()
plt.show()