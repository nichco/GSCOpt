import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from modopt import JaxProblem, SLSQP
import time
import jax
import jax.numpy as jnp 
jax.config.update("jax_enable_x64", True)

n = 30
dt = 2 / n
mc = 2
g = 9.81
d = 0.8
mu_cart = 0.03
mu_pole = 0.03
uscale = 10

x_init  = np.array([0, np.pi, 0, 0])
x_final = np.array([d, 0, 0, 0])



def jax_obj(v):
    u = v[n*4+2:].reshape((1, n)) * uscale

    j = 0.5 * dt * jnp.sum(u[0, :-1]**2 + u[0, 1:]**2)

    return 1e-2 * j



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

vl_x[:, 0] = x_init
vu_x[:, 0] = x_init

# final condition
vl_x[:, -1] = x_final
vu_x[:, -1] = x_final

# concatenate all bounds
vl = np.concatenate((np.array([vl_l, vl_mp]), vl_x.flatten(), vl_u))
vu = np.concatenate((np.array([vu_l, vu_mp]), vu_x.flatten(), vu_u))

nc = 4 * (n - 1)  # dynamics constraints



# initial guesses
q1_0 = np.linspace(0, d, n)
q2_0 = np.linspace(np.pi, 0, n)
q3_0 = np.zeros(n)
q4_0 = np.zeros(n)

u0 = np.zeros((n))

l0 = 0.5
mp0 = 0.4

x0 = np.concatenate((np.array([l0, mp0]), np.vstack((q1_0, q2_0, q3_0, q4_0)).flatten(), u0))




jaxprob = JaxProblem(x0=x0, nc=nc, jax_obj=jax_obj, jax_con=jax_con,
                    name=f'cart_pole_jax', order=1,
                    xl=vl, xu=vu, cl=0., cu=0.)





optimizer = SLSQP(jaxprob, solver_options={'maxiter': 500, 'ftol': 1e-9}, turn_off_outputs=True)
optimizer.solve()
optimizer.print_results()



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