import csdl_alpha as csdl
import numpy as np
from modopt import CSDLAlphaProblem
from modopt import SLSQP, IPOPT
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
import time

# 2
# objective:  [780.64659782]
# mp:  [0.57994239]
# length:  [0.37642662]

# 3
# objective:  [1169.81073232]
# mp:  [0.58082863]
# length:  [0.37607782]

# 4
# objective:  [1577.79095333]
# mp:  [0.581217]
# length:  [0.37146522]

# 5
# objective:  [1956.1952883]
# mp:  [0.57440568]
# length:  [0.38019928]

# 6
# objective:  [2335.2196999]
# mp:  [0.57385703]
# length:  [0.38268195]

# 7
# objective:  [2663.09899277]
# mp:  [0.56149822]
# length:  [0.40423861]

# 8
# objective:  [2920.92522549]
# mp:  [0.56564313]
# length:  [0.41534208]

# 9
# objective:  [3347.99159932]
# mp:  [0.53598422]
# length:  [0.44051492]

# 10:
# objective:  [3811.80963346]
# mp:  [0.51117499]
# length:  [0.45675482]


n = 30
dt = 2 / n
mc = 2
g = 9.81
d = 0.8
mu_cart = 0.03
mu_pole = 0.03


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

# initial_states = [initial_state_1, initial_state_2]
# initial_states = [initial_state_1, initial_state_2, initial_state_3]
# initial_states = [initial_state_1, initial_state_2, initial_state_3, initial_state_4]
# initial_states = [initial_state_1, initial_state_2, initial_state_3, initial_state_4, initial_state_5]
# initial_states = [initial_state_1, initial_state_2, initial_state_3, initial_state_4, initial_state_5, initial_state_6]
# initial_states = [initial_state_1, initial_state_2, initial_state_3, initial_state_4, initial_state_5, initial_state_6, initial_state_7]
# initial_states = [initial_state_1, initial_state_2, initial_state_3, initial_state_4, initial_state_5, initial_state_6, initial_state_7, initial_state_8]
# initial_states = [initial_state_1, initial_state_2, initial_state_3, initial_state_4, initial_state_5, initial_state_6, initial_state_7, initial_state_8, initial_state_9]
initial_states = [initial_state_1, initial_state_2, initial_state_3, initial_state_4, initial_state_5, initial_state_6, initial_state_7, initial_state_8, initial_state_9, initial_state_10]


N = len(initial_states)  # number of cp copies

recorder = csdl.Recorder(inline=True)
recorder.start()

j = 0

l = csdl.Variable(value=0.5)
l.set_as_design_variable(lower=0.1, upper=5, scaler=1)
l_hat = l / 2

mp = csdl.Variable(value=0.4)
mp.set_as_design_variable(lower=0.1, scaler=1)

x_data = []
for k in range(N):

    i_s = initial_states[k]

    q1_0 = np.linspace(0, d, n)
    q2_0 = np.linspace(np.pi, 0, n)
    # q1_0 = np.linspace(i_s[0], d, n)
    # q2_0 = np.linspace(i_s[1], 0, n)
    q3_0 = np.zeros(n)
    q4_0 = np.zeros(n)
    x = csdl.Variable(value=np.vstack((q1_0, q2_0, q3_0, q4_0)))
    x.set_as_design_variable(scaler=1)
    x_data.append(x)

    u = csdl.Variable(value=np.zeros((n)))
    u.set_as_design_variable(lower=-40, upper=40, scaler=1e-1)

    f = csdl.Variable(value=np.zeros((4, n)))
    for i in csdl.frange(n):

        cart_force = u[i]
        theta = x[1, :][i]
        dx = x[2, :][i]
        dtheta = x[3, :][i]

        ddx_num = mp * g * csdl.sin(theta) * csdl.cos(theta) - (7/3) * (cart_force + mp * l_hat * dtheta**2 * csdl.sin(theta) - mu_cart * dx) - (mu_pole * dtheta * csdl.cos(theta) / l_hat)
        ddx_den = mp * csdl.cos(theta)**2 - (7/3) * (mc + mp)
        ddx = ddx_num / ddx_den

        ddtheta = 3 * (g * csdl.sin(theta) - ddx * csdl.cos(theta) - (mu_pole * dtheta / (mp * l_hat))) / (7 * l_hat)

        f = f.set(csdl.slice[0, i], x[2, i])
        f = f.set(csdl.slice[1, i], x[3, i])
        f = f.set(csdl.slice[2, i], ddx)
        f = f.set(csdl.slice[3, i], ddtheta)

    r = csdl.Variable(value=np.zeros((4, n - 1)))
    for i in csdl.frange(n - 1):
        r = r.set(csdl.slice[:, i], x[:, i+1] - x[:, i] - 0.5 * dt * (f[:, i+1] + f[:, i]))

    r.set_as_constraint(equals=0, scaler=1E1)

    x[:, 0].set_as_constraint(equals=i_s, scaler=1)

    x[1, :].set_as_constraint(lower=-1e-3, scaler=1) # lower bound on angle
    x[0, :].set_as_constraint(lower=i_s[0]-1e-3, scaler=1)  # lower bound on cart position

    x[:, n - 1].set_as_constraint(equals=np.array([d, 0, 0, 0]), scaler=1)


    for i in range(n - 1):
        j = j + 0.5 * dt * (u[i]**2 + u[i + 1]**2)




j.set_as_objective(scaler=1E-2)


recorder.stop()


sim = csdl.experimental.JaxSimulator(recorder=recorder)
prob = CSDLAlphaProblem(simulator=sim)
t1 = time.time()
optimizer = SLSQP(prob, solver_options={'maxiter': 3000, 'ftol': 1e-7}, turn_off_outputs=True)
optimizer.solve()
optimizer.print_results()
t2 = time.time()
print('Time (s): ', t2 - t1)


print('objective: ', j.value)
print('mp: ', mp.value)
print('length: ', l.value)














x_data = [xx.value for xx in x_data]
l_data = [l.value for i in range(N)]

# plot the cart-pole trajectories
fig, axs = plt.subplots(2, 5, figsize=(17, 6))
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