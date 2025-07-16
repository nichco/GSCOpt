import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

# filename2 = 'relaxation/convergence/monolithic_2.out'
# filename3 = 'relaxation/convergence/monolithic_3.out'
# filename4 = 'relaxation/convergence/monolithic_4.out'
# filename5 = 'relaxation/convergence/monolithic_5.out'
# filename6 = 'relaxation/convergence/monolithic_6.out'
# filename7 = 'relaxation/convergence/monolithic_7.out'
# filename8 = 'relaxation/convergence/monolithic_8.out'
# filename9 = 'relaxation/convergence/monolithic_9.out'
# filename10 = 'relaxation/convergence/monolithic_10.out'
filename2 = 'relaxation/convergence/m2.out'
filename3 = 'relaxation/convergence/m3.out'
filename4 = 'relaxation/convergence/m4.out'
filename5 = 'relaxation/convergence/m5.out'
filename6 = 'relaxation/convergence/m6.out'
filename7 = 'relaxation/convergence/m7.out'
filename8 = 'relaxation/convergence/m8.out'
filename9 = 'relaxation/convergence/m9.out'
filename10 = 'relaxation/convergence/m10.out'

# Read the header line separately
with open(filename2, 'r') as file:
    headers2 = file.readline().strip().split()

with open(filename3, 'r') as file:
    headers3 = file.readline().strip().split()

with open(filename4, 'r') as file:
    headers4 = file.readline().strip().split()

with open(filename5, 'r') as file:
    headers5 = file.readline().strip().split()

with open(filename6, 'r') as file:
    headers6 = file.readline().strip().split()

with open(filename7, 'r') as file:
    headers7 = file.readline().strip().split()

with open(filename8, 'r') as file:
    headers8 = file.readline().strip().split()

with open(filename9, 'r') as file:
    headers9 = file.readline().strip().split()

with open(filename10, 'r') as file:
    headers10 = file.readline().strip().split()

# Read the rest of the data into a DataFrame
m2 = pd.read_csv(filename2, delim_whitespace=True, skiprows=1, names=headers2)
m3 = pd.read_csv(filename3, delim_whitespace=True, skiprows=1, names=headers3)
m4 = pd.read_csv(filename4, delim_whitespace=True, skiprows=1, names=headers4)
m5 = pd.read_csv(filename5, delim_whitespace=True, skiprows=1, names=headers5)
m6 = pd.read_csv(filename6, delim_whitespace=True, skiprows=1, names=headers6)
m7 = pd.read_csv(filename7, delim_whitespace=True, skiprows=1, names=headers7)
m8 = pd.read_csv(filename8, delim_whitespace=True, skiprows=1, names=headers8)
m9 = pd.read_csv(filename9, delim_whitespace=True, skiprows=1, names=headers9)
m10 = pd.read_csv(filename10, delim_whitespace=True, skiprows=1, names=headers10)

m2_major = m2['MAJOR'].to_numpy()
m2_opt = m2['OPT'].to_numpy()
m2_feas = m2['FEAS'].to_numpy()
m2_obj = m2['OBJFUN'].to_numpy()

m3_major = m3['MAJOR'].to_numpy()
m3_opt = m3['OPT'].to_numpy()
m3_feas = m3['FEAS'].to_numpy()
m3_obj = m3['OBJFUN'].to_numpy()

m4_major = m4['MAJOR'].to_numpy()
m4_opt = m4['OPT'].to_numpy()
m4_feas = m4['FEAS'].to_numpy()
m4_obj = m4['OBJFUN'].to_numpy()

m5_major = m5['MAJOR'].to_numpy()
m5_opt = m5['OPT'].to_numpy()
m5_feas = m5['FEAS'].to_numpy()
m5_obj = m5['OBJFUN'].to_numpy()

m6_major = m6['MAJOR'].to_numpy()
m6_opt = m6['OPT'].to_numpy()
m6_feas = m6['FEAS'].to_numpy()
m6_obj = m6['OBJFUN'].to_numpy()

m7_major = m7['MAJOR'].to_numpy()
m7_opt = m7['OPT'].to_numpy()
m7_feas = m7['FEAS'].to_numpy()
m7_obj = m7['OBJFUN'].to_numpy()

m8_major = m8['MAJOR'].to_numpy()
m8_opt = m8['OPT'].to_numpy()
m8_feas = m8['FEAS'].to_numpy()
m8_obj = m8['OBJFUN'].to_numpy()

m9_major = m9['MAJOR'].to_numpy()
m9_opt = m9['OPT'].to_numpy()
m9_feas = m9['FEAS'].to_numpy()
m9_obj = m9['OBJFUN'].to_numpy()

m10_major = m10['MAJOR'].to_numpy()
m10_opt = m10['OPT'].to_numpy()
m10_feas = m10['FEAS'].to_numpy()
m10_obj = m10['OBJFUN'].to_numpy()

# plt.plot(m2_major, m2_obj, label='Monolithic 2 copies', color='tab:blue')
# plt.plot(m3_major, m3_obj, label='Monolithic 3 copies', color='tab:orange')
# plt.plot(m4_major, m4_obj, label='Monolithic 4 copies', color='tab:green')
# plt.plot(m5_major, m5_obj, label='Monolithic 5 copies', color='tab:red')
# plt.plot(m6_major, m6_obj, label='Monolithic 6 copies', color='tab:purple')
# plt.plot(m7_major, m7_obj, label='Monolithic 7 copies', color='tab:brown')
# plt.plot(m8_major, m8_obj, label='Monolithic 8 copies', color='tab:gray')
# plt.plot(m9_major, m9_obj, label='Monolithic 9 copies', color='tab:pink')
# plt.plot(m10_major, m10_obj, label='Monolithic 10 copies', color='tab:cyan')
plt.loglog(m2_major, m2_obj, label='Monolithic 2 copies', color='tab:blue', linewidth=3)
plt.loglog(m3_major, m3_obj, label='Monolithic 3 copies', color='tab:orange', linewidth=3)
plt.loglog(m4_major, m4_obj, label='Monolithic 4 copies', color='tab:green', linewidth=3)
plt.loglog(m5_major, m5_obj, label='Monolithic 5 copies', color='tab:red', linewidth=3)
plt.loglog(m6_major, m6_obj, label='Monolithic 6 copies', color='tab:purple', linewidth=3)
plt.loglog(m7_major, m7_obj, label='Monolithic 7 copies', color='tab:brown', linewidth=3)
plt.loglog(m8_major, m8_obj, label='Monolithic 8 copies', color='tab:gray', linewidth=3)
plt.loglog(m9_major, m9_obj, label='Monolithic 9 copies', color='tab:pink', linewidth=3)
plt.loglog(m10_major, m10_obj, label='Monolithic 10 copies', color='tab:cyan', linewidth=3)
# plt.plot(m2_major, m2_opt, label='Monolithic 2 copies', color='tab:blue', linewidth=3)
# plt.plot(m3_major, m3_opt, label='Monolithic 3 copies', color='tab:orange', linewidth=3)
# plt.plot(m4_major, m4_opt, label='Monolithic 4 copies', color='tab:green', linewidth=3)
# plt.plot(m5_major, m5_opt, label='Monolithic 5 copies', color='tab:red', linewidth=3)
# plt.plot(m6_major, m6_opt, label='Monolithic 6 copies', color='tab:purple', linewidth=3)
# plt.plot(m7_major, m7_opt, label='Monolithic 7 copies', color='tab:brown', linewidth=3)
# plt.plot(m8_major, m8_opt, label='Monolithic 8 copies', color='tab:gray', linewidth=3)
# plt.plot(m9_major, m9_opt, label='Monolithic 9 copies', color='tab:pink', linewidth=3)
# plt.plot(m10_major, m10_opt, label='Monolithic 10 copies', color='tab:cyan', linewidth=3)
plt.xlabel('Iteration')
# plt.ylabel('Optimality')
plt.ylabel('Objective function value')
plt.legend()
plt.yscale('log')
plt.grid(color='lavender', alpha=0.5)
# plt.ylim(1e1, 1e2)
plt.show()







# open obj2 pickle file
with open('relaxation/convergence/obj2.pkl', 'rb') as f:
    obj2 = pickle.load(f)

with open('relaxation/convergence/obj3.pkl', 'rb') as f:
    obj3 = pickle.load(f)

with open('relaxation/convergence/obj4.pkl', 'rb') as f:
    obj4 = pickle.load(f)

with open('relaxation/convergence/obj5.pkl', 'rb') as f:
    obj5 = pickle.load(f)

with open('relaxation/convergence/obj6.pkl', 'rb') as f:
    obj6 = pickle.load(f)

with open('relaxation/convergence/obj7.pkl', 'rb') as f:
    obj7 = pickle.load(f)

with open('relaxation/convergence/obj8.pkl', 'rb') as f:
    obj8 = pickle.load(f)

with open('relaxation/convergence/obj9.pkl', 'rb') as f:
    obj9 = pickle.load(f)

with open('relaxation/convergence/obj10.pkl', 'rb') as f:
    obj10 = pickle.load(f)


# plt.plot(obj2[1:], linewidth=3)
# plt.plot(obj3[2:], linewidth=3)
# plt.plot(obj4[3:], linewidth=3)
# plt.plot(obj5[4:], linewidth=3)
# plt.plot(obj6[5:], linewidth=3)
# # plt.plot(obj7[6:], linewidth=3)
# plt.plot(obj8[7:], linewidth=3)
# plt.plot(obj9[8:], linewidth=3)
# plt.plot(obj10[9:], linewidth=3)

# plt.plot(obj2[0:], linewidth=3)
# plt.plot(obj3[1:], linewidth=3)
# plt.plot(obj4[2:], linewidth=3)
# plt.plot(obj5[3:], linewidth=3)
# plt.plot(obj6[4:], linewidth=3)
# plt.plot(obj7[5:], linewidth=3)
# plt.plot(obj8[6:], linewidth=3)
# plt.plot(obj9[7:], linewidth=3)
# plt.plot(obj10[8:], linewidth=3)

plt.plot(obj2, linewidth=3)
plt.plot(obj3, linewidth=3)
plt.plot(obj4, linewidth=3)
plt.plot(obj5, linewidth=3)
plt.plot(obj6, linewidth=3)
plt.plot(obj7, linewidth=3)
plt.plot(obj8, linewidth=3)
plt.plot(obj9, linewidth=3)
plt.plot(obj10, linewidth=3)

plt.xlabel('Iteration')
plt.ylabel('Objective function value')
plt.show()