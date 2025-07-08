import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from modopt import JaxProblem, SLSQP
import time
import jax
import jax.numpy as jnp 
jax.config.update("jax_enable_x64", True)
import tracemalloc

n = 30
dt = 2 / n
mc = 2
g = 9.81
d = 0.8
mu_cart = 0.03
mu_pole = 0.03
uscale = 10

# x_init  = np.array([0, np.pi, 0, 0])
# x_final = np.array([d, 0, 0, 0])

initial_state_1 = np.array([0, np.pi, 0, 0])
initial_state_2 = np.array([0.5, np.pi, 0, 0])
initial_state_3 = np.array([0.25, np.pi, 0, 0])
initial_state_4 = np.array([0.75, np.pi, 0, 0])
initial_state_5 = np.array([-0.5, np.pi, 0, 0])
initial_state_6 = np.array([-0.25, np.pi, 0, 0])
initial_state_7 = np.array([0, np.pi+np.pi/2, 0, 0])
initial_state_8 = np.array([0.3, np.pi+np.pi/2, 0, 0])
initial_state_9 = np.array([-0.5, np.pi+np.pi/2, 0, 0])
initial_state_10 = np.array([-3, np.pi, 0, 0])

# initial_states = [initial_state_1]
# initial_states = [initial_state_1, initial_state_2]
# initial_states = [initial_state_1, initial_state_2, initial_state_3]
# initial_states = [initial_state_1, initial_state_2, initial_state_3, initial_state_4]
# initial_states = [initial_state_1, initial_state_2, initial_state_3, initial_state_4, initial_state_5]
# initial_states = [initial_state_1, initial_state_2, initial_state_3, initial_state_4, initial_state_5, initial_state_6]
# initial_states = [initial_state_1, initial_state_2, initial_state_3, initial_state_4, initial_state_5, initial_state_6, initial_state_7]
# initial_states = [initial_state_1, initial_state_2, initial_state_3, initial_state_4, initial_state_5, initial_state_6, initial_state_7, initial_state_8]
# initial_states = [initial_state_1, initial_state_2, initial_state_3, initial_state_4, initial_state_5, initial_state_6, initial_state_7, initial_state_8, initial_state_9]
initial_states = [initial_state_1, initial_state_2, initial_state_3, initial_state_4, initial_state_5, initial_state_6, initial_state_7, initial_state_8, initial_state_9, initial_state_10]


N = len(initial_states)

# keep track of indices in v
vpb = 2 + n*5 # variables per block
# l_i = i * vpb
# mp_i = i * vpb + 1
# x_i = i * vpb + 2
# x_f = i * vpb + 2 + n*4
# u_i = i * vpb + 2 + n*4
# u_f = i * vpb + 2 + n*4 + n


def jax_obj(v):

    j = 0

    for i in range(N):
        ui = v[i * vpb + 2 + n*4 : i * vpb + 2 + n*4 + n].reshape((1, n)) * uscale
        ji = 0.5 * dt * jnp.sum(ui[0, :-1]**2 + ui[0, 1:]**2)
        j = j + ji

    # u = v[n*4+2:].reshape((1, n)) * uscale
    # j = 0.5 * dt * jnp.sum(u[0, :-1]**2 + u[0, 1:]**2)

    return 1e-2 * j



def jax_con(v):

    c = []

    for i in range(N):
        li = v[i * vpb]
        li_hat = li / 2
        mpi = v[i * vpb + 1]
        xi = v[i*vpb + 2 : i * vpb + 2 + n*4].reshape((4, n))
        ui = v[i * vpb + 2 + n*4 : i * vpb + 2 + n*4 + n].reshape((n)) * uscale

        theta_i = xi[1, :]
        dx_i = xi[2, :]
        dtheta_i = xi[3, :]
        cart_force_i = ui

        sin_theta_i = jnp.sin(theta_i)
        cos_theta_i = jnp.cos(theta_i)

        ddx_num_i = (
                mpi * g * sin_theta_i * cos_theta_i
                - (7 / 3) * (cart_force_i + mpi * li_hat * dtheta_i**2 * sin_theta_i - mu_cart * dx_i)
                - (mu_pole * dtheta_i * cos_theta_i / li_hat)
            )
        
        ddx_den_i = mpi * cos_theta_i**2 - (7 / 3) * (mc + mpi)
        ddx_i = ddx_num_i / ddx_den_i

        ddtheta_i = 3 * (g * sin_theta_i - ddx_i * cos_theta_i - (mu_pole * dtheta_i / (mpi * li_hat))) / (7 * li_hat)

        # Stack all into f: shape (4, n)
        fi = jnp.vstack((dx_i, dtheta_i, ddx_i, ddtheta_i))

        ci = xi[:, 1:] - xi[:, :-1] - 0.5 * dt * (fi[:, 1:] + fi[:, :-1])
        c.append(ci.flatten())

    # l = v[0]
    # l_hat = l / 2
    # mp = v[1]
    # x = v[2:n*4+2].reshape((4, n))
    # u = v[n*4+2:].reshape((n)) * uscale

    # theta = x[1, :]
    # dx = x[2, :]
    # dtheta = x[3, :]
    # cart_force = u

    # sin_theta = jnp.sin(theta)
    # cos_theta = jnp.cos(theta)

    # ddx_num = (
    #     mp * g * sin_theta * cos_theta
    #     - (7 / 3) * (cart_force + mp * l_hat * dtheta**2 * sin_theta - mu_cart * dx)
    #     - (mu_pole * dtheta * cos_theta / l_hat)
    # )

    # ddx_den = mp * cos_theta**2 - (7 / 3) * (mc + mp)
    # ddx = ddx_num / ddx_den

    # ddtheta = 3 * (g * sin_theta - ddx * cos_theta - (mu_pole * dtheta / (mp * l_hat))) / (7 * l_hat)

    # # Stack all into f: shape (4, n)
    # f = jnp.vstack((dx, dtheta, ddx, ddtheta))


    # c = x[:, 1:] - x[:, :-1] - 0.5 * dt * (f[:, 1:] + f[:, :-1])
    
    # return 10 * c.flatten()

    return 10 * jnp.concatenate(c)



vl = []
vu = []

for i in range(N):

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
    vl_i = np.concatenate((np.array([0.1, 0.1]), vl_x.flatten(), vl_u))
    vl.append(vl_i)
    vu_i = np.concatenate((np.array([5.0, np.inf]), vu_x.flatten(), vu_u))
    vu.append(vu_i)




nc = N * (4 * (n - 1))  # dynamics constraints


vl = np.vstack(vl).flatten()
vu = np.vstack(vu).flatten()


# initial guesses
q1_0 = np.linspace(0, d, n)
q2_0 = np.linspace(np.pi, 0, n)
q3_0 = np.zeros(n)
q4_0 = np.zeros(n)

u0 = np.zeros((n))

l0 = 0.5
mp0 = 0.4

x0 = np.concatenate((np.array([l0, mp0]), np.vstack((q1_0, q2_0, q3_0, q4_0)).flatten(), u0))
x0 = np.vstack([x0 for _ in range(N)]).flatten()

tracemalloc.start()

jaxprob = JaxProblem(x0=x0, nc=nc, jax_obj=jax_obj, jax_con=jax_con,
                    name=f'cart_pole_jax', order=1,
                    xl=vl, xu=vu, cl=0., cu=0.)





optimizer = SLSQP(jaxprob, solver_options={'maxiter': 3000, 'ftol': 1e-7}, turn_off_outputs=True)
optimizer.solve()
optimizer.print_results()



# print peak memory usage
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"Peak memory usage: {peak / 1e6} MB")



exit()

v = optimizer.results['x']
l = v[0]
mp = v[1]
x = v[2:n*4+2].reshape((4, n))
u = v[n*4+2:].reshape((1, n))


print('l: ', l)
print('mp: ', mp)







t = np.linspace(0, n*dt, n)
position = x[0, :]  # cart position
angle = x[1, :]    # pole angle


plt.plot(position)
plt.plot(angle)
plt.plot(u.flatten())
plt.show()




# Compute pole tip position
pole_x = position + l * np.sin(angle)
pole_y = l * np.cos(angle)



# Animation setup
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.grid(True)

# Determine plot limits
x_min, x_max = position.min() - l - 0.5, position.max() + l + 0.5
y_min, y_max = -l - 0.2, l + 0.2
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Cart parameters
cart_width = 0.4
cart_height = 0.2

# Create cart as a Rectangle patch
cart = plt.Rectangle((position[0] - cart_width/2, -cart_height/2), cart_width, cart_height, facecolor='lightblue')
ax.add_patch(cart)

# Create pole as a line
pole_line, = ax.plot([], [], lw=5, color='red')
pole_tip, = ax.plot([], [], 'o', color='green', markersize=12)

def animate(i):
    # Update cart position
    cart.set_xy((position[i] - cart_width/2, -cart_height/2))
    # Update pole line: from cart center up to pole tip
    x0, y0 = position[i], 0
    pole_line.set_data([x0, pole_x[i]], [y0, pole_y[i]])
    pole_tip.set_data([pole_x[i]], [pole_y[i]])
    return cart, pole_line, pole_tip

# Create animation
ani = FuncAnimation(fig, animate, frames=len(t), blit=True, interval=30)

# from matplotlib.animation import PillowWriter
# ani.save("cart_pole_animation.gif", writer=PillowWriter(fps=30))

plt.show()




# results:
# obj: 3.8284262931068933
# l:  0.3870413653072123
# mp:  0.5760740845717346

# distributed:
# obj: 2 * 3.828426293192311
# l2:  0.38703993961272476
# mp2:  0.5760724540187763