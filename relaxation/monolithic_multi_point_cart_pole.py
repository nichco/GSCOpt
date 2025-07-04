import csdl_alpha as csdl
import numpy as np
from modopt import CSDLAlphaProblem
from modopt import SLSQP, IPOPT
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# np.random.seed(0)
# initial_position = np.random.uniform(-2, 2, 100)  # Seed for reproducibility

# dynamics from https://sharpneat.sourceforge.io/research/cart-pole/cart-pole-equations.html
# co-design problem by me

n = 30
dt = 2 / n
mc = 2
g = 9.81
d = 0.8
mu_cart = 0.03
mu_pole = 0.03

initial_state_1 = np.array([0, np.pi, 0, 0])
initial_state_2 = np.array([1, np.pi, 0, 0])
initial_state_3 = np.array([0.5, np.pi, 0, 0])
initial_state_4 = np.array([0.75, np.pi, 0, 0])
initial_states = [initial_state_1, initial_state_2, initial_state_3, initial_state_4]


recorder = csdl.Recorder(inline=True)
recorder.start()

j = 0

l = csdl.Variable(value=0.5)
l.set_as_design_variable(lower=0.1, upper=5, scaler=1)
l_hat = l / 2

mp = csdl.Variable(value=0.4)
mp.set_as_design_variable(lower=0.1, scaler=1)


for k in range(4):

    q1_0 = np.linspace(0, d, n)
    q2_0 = np.linspace(np.pi, 0, n)
    q3_0 = np.zeros(n)
    q4_0 = np.zeros(n)
    x = csdl.Variable(value=np.vstack((q1_0, q2_0, q3_0, q4_0)))
    x.set_as_design_variable(scaler=1)

    u = csdl.Variable(value=np.zeros((n)))
    u.set_as_design_variable(lower=-50, upper=50, scaler=1e-1)

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

    # x[:, 0].set_as_constraint(equals=np.array([0, np.pi, 0, 0]), scaler=1)
    x[:, 0].set_as_constraint(equals=initial_states[k], scaler=1)

    x[:, n - 1].set_as_constraint(equals=np.array([d, 0, 0, 0]), scaler=1)


    for i in range(n - 1):
        j = j + 0.5 * dt * (u[i]**2 + u[i + 1]**2)




j.set_as_objective(scaler=1E-2)


recorder.stop()


sim = csdl.experimental.JaxSimulator(recorder=recorder)
prob = CSDLAlphaProblem(simulator=sim)
optimizer = SLSQP(prob, solver_options={'maxiter': 1000, 'ftol': 1e-9}, turn_off_outputs=True)
optimizer.solve()
optimizer.print_results()


print('objective: ', j.value)
print('mp: ', mp.value)
print('length: ', l.value)



# optimal design variables with two copies
# obj: 8.03710769545161
# mp:  0.57547296
# length:  0.36931781


# optimal design variables with three copies
# objective:  12.0105464351
# mp:  0.57870551
# length:  0.36804107


# optimal design variables with four copies
# objective:  16.0860669215
# mp:  0.57991946
# length:  0.36539453

# distributed results with two copies
# obj: 8.03710769555307
# mp: 0.5754734938307297
# length: 0.3693175514897895


# distributed results with three copies
# objective:  12.010546218062135
# mp:  0.5787555766374402
# l:  0.3680019980172635



# distributed results with four copies
# Objective:  16.063395146305165
# l1:  0.3110872760510847
# mp1:  0.6837186937953063
# l2:  0.3110920497136964
# mp2:  0.6837031836555887
# l3:  0.31109741579491706
# mp3:  0.6836869338140287
# l4:  0.311101440251502
# mp4:  0.6836647838678316