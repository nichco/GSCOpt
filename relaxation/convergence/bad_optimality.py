import numpy as np
import matplotlib.pyplot as plt
import pickle
import csdl_alpha as csdl
from modopt import CSDLAlphaProblem
from modopt import SLSQP, IPOPT, PySLSQP


N = 2


def prob(v_init: list):


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

    initial_states = [initial_state_1, initial_state_2]
    # initial_states = [initial_state_1, initial_state_2, initial_state_3]
    # initial_states = [initial_state_1, initial_state_2, initial_state_3, initial_state_4]
    # initial_states = [initial_state_1, initial_state_2, initial_state_3, initial_state_4, initial_state_5]
    # initial_states = [initial_state_1, initial_state_2, initial_state_3, initial_state_4, initial_state_5, initial_state_6]
    # initial_states = [initial_state_1, initial_state_2, initial_state_3, initial_state_4, initial_state_5, initial_state_6, initial_state_7]
    # initial_states = [initial_state_1, initial_state_2, initial_state_3, initial_state_4, initial_state_5, initial_state_6, initial_state_7, initial_state_8]
    # initial_states = [initial_state_1, initial_state_2, initial_state_3, initial_state_4, initial_state_5, initial_state_6, initial_state_7, initial_state_8, initial_state_9]
    # initial_states = [initial_state_1, initial_state_2, initial_state_3, initial_state_4, initial_state_5, initial_state_6, initial_state_7, initial_state_8, initial_state_9, initial_state_10]


    N = len(initial_states)  # number of cp copies

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    j = 0

    l = csdl.Variable(value=v_init[0]) # ************************************************************************************************
    l.set_as_design_variable(lower=0.1, upper=5, scaler=1)
    l_hat = l / 2

    mp = csdl.Variable(value=v_init[1]) # ************************************************************************************************
    mp.set_as_design_variable(lower=0.1, scaler=1)

    x_data = []
    for k in range(N):

        i_s = initial_states[k]

        x = csdl.Variable(value=v_init[2][k]) # ************************************************************************************************
        x.set_as_design_variable(scaler=1)
        x_data.append(x)

        u = csdl.Variable(value=v_init[3][k]) # ************************************************************************************************
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
    # optimizer = SLSQP(prob, solver_options={'maxiter': 3000, 'ftol': 1e-7}, turn_off_outputs=True)
    optimizer = PySLSQP(prob, solver_options={'maxiter': 2, 'acc': 1e-7}, turn_off_outputs=True)
    optimizer.solve()
    optimizer.print_results()

    optimality = optimizer.results['optimality']

    return optimality


# open h2.pkl file
with open('relaxation/convergence/h2.pkl', 'rb') as f:
    h2 = pickle.load(f)

opt = []

for i, v in enumerate(h2):

    l_list = []
    mp_list = []
    x_list = []
    u_list = []

    # l1 = v_init[0]
    # l2 = v_init[1]
    # mp1 = v_init[2]
    # mp2 = v_init[3]
    # x1 = v_init[4]
    # x2 = v_init[5]
    # u1 = v_init[6]
    # u2 = v_init[7]

    for j in range(N):
        l_list.append(v[j])
        mp_list.append(v[j + N])
        x_list.append(v[j + 2 * N])
        u_list.append(v[j + 3 * N])

    k = (i % N) # the last block index
    # print(k)

    # print(l_list)
    # print(mp_list)
    # print(x_list)
    # print(u_list)
    # exit()


    v_init = [l_list[k], mp_list[k], x_list, u_list]


    opt.append(prob(v_init))


    if i > 50: break




plt.plot(opt)
plt.xlabel('Iteration')
plt.ylabel('Optimality')
plt.show()