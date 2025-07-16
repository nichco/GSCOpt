import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

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

tm2 = np.linspace(0, 9.86, len(m2_major))
tm3 = np.linspace(0, 24.22, len(m3_major))
tm4 = np.linspace(0, 112.345, len(m4_major))
tm5 = np.linspace(0, 386.23, len(m5_major))
tm6 = np.linspace(0, 755.009, len(m6_major))
tm7 = np.linspace(0, 1504.773, len(m7_major))
tm8 = np.linspace(0, 2785.930, len(m8_major))
tm9 = np.linspace(0, 3508.666, len(m9_major))
tm10 = np.linspace(0, 15755.439, len(m10_major))

td2 = np.linspace(0, 40.675, len(obj2))
td3 = np.linspace(0, 58.815, len(obj3))
td4 = np.linspace(0, 176.175, len(obj4))
td5 = np.linspace(0, 265.7, len(obj5))
td6 = np.linspace(0, 320.679, len(obj6))
td7 = np.linspace(0, 399.415, len(obj7))
td8 = np.linspace(0, 565.788, len(obj8))
td9 = np.linspace(0, 588.668, len(obj9))
td10 = np.linspace(0, 817.812, len(obj10))




f2 = m2_obj[-1]
f3 = m3_obj[-1]
f4 = m4_obj[-1]
f5 = m5_obj[-1]
f6 = m6_obj[-1]
f7 = m7_obj[-1]
f8 = m8_obj[-1]
f9 = m9_obj[-1]
f10 = m10_obj[-1]




plt.figure(figsize=(7, 5))
plt.rcParams.update({'font.size': 14})


plt.loglog(tm2, np.abs(m2_obj - f2), label='_nolegend_', linewidth=3, color='tab:blue', zorder=10)
plt.loglog(td2, np.abs(obj2 - f2), label='_nolegend_', linewidth=3, color='tab:blue', linestyle='--', zorder=10)

plt.loglog(tm3, np.abs(m3_obj - f3), label='_nolegend_', linewidth=3, color='tab:orange', zorder=9)
plt.loglog(td3, np.abs(obj3 - f3), label='_nolegend_', linewidth=3, color='tab:orange', linestyle='--', zorder=9)

plt.loglog(tm4, np.abs(m4_obj - f4), label='_nolegend_', linewidth=3, color='tab:green', zorder=8)
plt.loglog(td4, np.abs(obj4 - f4), label='_nolegend_', linewidth=3, color='tab:green', linestyle='--', zorder=8)

plt.loglog(tm5, np.abs(m5_obj - f5), label='_nolegend_', linewidth=3, color='tab:red', zorder=7)
plt.loglog(td5, np.abs(obj5 - f5), label='_nolegend_', linewidth=3, color='tab:red', linestyle='--', zorder=7)

plt.loglog(tm6, np.abs(m6_obj - f6), label='_nolegend_', linewidth=3, color='tab:purple', zorder=6)
plt.loglog(td6, np.abs(obj6 - f6), label='_nolegend_', linewidth=3, color='tab:purple', linestyle='--', zorder=6)

plt.loglog(tm7, np.abs(m7_obj - f7), label='_nolegend_', linewidth=3, color='tab:brown', zorder=5)
plt.loglog(td7, np.abs(obj7 - f7), label='_nolegend_', linewidth=3, color='tab:brown', linestyle='--', zorder=5)

plt.loglog(tm8, np.abs(m8_obj - f8), label='_nolegend_', linewidth=3, color='tab:gray', zorder=4)
plt.loglog(td8, np.abs(obj8 - f8), label='_nolegend_', linewidth=3, color='tab:gray', linestyle='--', zorder=4)

plt.loglog(tm9, np.abs(m9_obj - f9), label='_nolegend_', linewidth=3, color='tab:pink', zorder=3)
plt.loglog(td9, np.abs(obj9 - f9), label='_nolegend_', linewidth=3, color='tab:pink', linestyle='--', zorder=3)

plt.loglog(tm10, np.abs(m10_obj - f10), label='_nolegend_', linewidth=3, color='tab:cyan', zorder=2)
plt.loglog(td10, np.abs(obj10 - f10), label='_nolegend_', linewidth=3, color='tab:cyan', linestyle='--', zorder=2)



plt.scatter(td2[-1], np.abs(obj2[-1] - f2), color='tab:blue', marker='o', s=120, label='2', linewidths=2, edgecolors='black', zorder=10,)
plt.scatter(td3[-1], np.abs(obj3[-1] - f3), color='tab:orange', marker='s', s=120, label='3', linewidths=2, edgecolors='black', zorder=10,)
plt.scatter(td4[-1], np.abs(obj4[-1] - f4), color='tab:green', marker='^', s=120, label='4', linewidths=2, edgecolors='black', zorder=10,)
plt.scatter(td5[-1], np.abs(obj5[-1] - f5), color='tab:red', marker='p', s=120, label='5', linewidths=2, edgecolors='black', zorder=10,)
plt.scatter(td6[-1], np.abs(obj6[-1] - f6), color='tab:purple', marker='>', s=120, label='6', linewidths=2, edgecolors='black', zorder=10,)
plt.scatter(td7[-1], np.abs(obj7[-1] - f7), color='tab:brown', marker='D', s=120, label='7', linewidths=2, edgecolors='black', zorder=10,)
plt.scatter(td8[-1], np.abs(obj8[-1] - f8), color='tab:gray', marker='<', s=120, label='8', linewidths=2, edgecolors='black', zorder=10,)
plt.scatter(td9[-1], np.abs(obj9[-1] - f9), color='tab:pink', marker='P', s=120, label='9', linewidths=2, edgecolors='black', zorder=10,)
plt.scatter(td10[-1], np.abs(obj10[-1] - f10), color='tab:cyan', marker='v', s=120, label='10', linewidths=2, edgecolors='black', zorder=10,)



plt.ylabel('Objective function error')
plt.xlabel('CPU Time (s)')
leg = plt.legend(fontsize=12,)
leg.zorder = 11
plt.savefig('convergence.png', dpi=300, transparent=True, bbox_inches='tight')
plt.show()


exit()


plt.figure(figsize=(5, 3))
plt.rcParams.update({'font.size': 12})

# plt.plot(m2_major, m2_obj, label='Monolithic 2 copies', color='tab:blue')
# plt.plot(m3_major, m3_obj, label='Monolithic 3 copies', color='tab:orange')
# plt.plot(m4_major, m4_obj, label='Monolithic 4 copies', color='tab:green')
# plt.plot(m5_major, m5_obj, label='Monolithic 5 copies', color='tab:red')
# plt.plot(m6_major, m6_obj, label='Monolithic 6 copies', color='tab:purple')
# plt.plot(m7_major, m7_obj, label='Monolithic 7 copies', color='tab:brown')
# plt.plot(m8_major, m8_obj, label='Monolithic 8 copies', color='tab:gray')
# plt.plot(m9_major, m9_obj, label='Monolithic 9 copies', color='tab:pink')
# plt.plot(m10_major, m10_obj, label='Monolithic 10 copies', color='tab:cyan')

# plt.loglog(m2_major, m2_obj, label='Monolithic 2 copies', color='tab:blue', linewidth=3)
# plt.loglog(m3_major, m3_obj, label='Monolithic 3 copies', color='tab:orange', linewidth=3)
# plt.loglog(m4_major, m4_obj, label='Monolithic 4 copies', color='tab:green', linewidth=3)
# plt.loglog(m5_major, m5_obj, label='Monolithic 5 copies', color='tab:red', linewidth=3)
# plt.loglog(m6_major, m6_obj, label='Monolithic 6 copies', color='tab:purple', linewidth=3)
# plt.loglog(m7_major, m7_obj, label='Monolithic 7 copies', color='tab:brown', linewidth=3)
# plt.loglog(m8_major, m8_obj, label='Monolithic 8 copies', color='tab:gray', linewidth=3)
# plt.loglog(m9_major, m9_obj, label='Monolithic 9 copies', color='tab:pink', linewidth=3)
# plt.loglog(m10_major, m10_obj, label='Monolithic 10 copies', color='tab:cyan', linewidth=3)

# plt.loglog(m2_major, m2_opt, label='Monolithic 2', color='tab:blue', linewidth=3)
# plt.loglog(m3_major, m3_opt, label='Monolithic 3', color='tab:orange', linewidth=3)
# plt.loglog(m4_major, m4_opt, label='Monolithic 4', color='tab:green', linewidth=3)
# plt.loglog(m5_major, m5_opt, label='Monolithic 5', color='tab:red', linewidth=3)
# plt.loglog(m6_major, m6_opt, label='Monolithic 6', color='tab:purple', linewidth=3)
# plt.loglog(m7_major, m7_opt, label='Monolithic 7', color='tab:brown', linewidth=3)
# plt.loglog(m8_major, m8_opt, label='Monolithic 8', color='tab:gray', linewidth=3)
# plt.loglog(m9_major, m9_opt, label='Monolithic 9', color='tab:pink', linewidth=3)
# plt.loglog(m10_major, m10_opt, label='Monolithic 10', color='tab:cyan', linewidth=3)

# plt.plot(m2_major, m2_opt, label='Monolithic 2', color='tab:blue', linewidth=3)
# plt.plot(m3_major, m3_opt, label='Monolithic 3', color='tab:orange', linewidth=3)
# plt.plot(m4_major, m4_opt, label='Monolithic 4', color='tab:green', linewidth=3)
# plt.plot(m5_major, m5_opt, label='Monolithic 5', color='tab:red', linewidth=3)
# plt.plot(m6_major, m6_opt, label='Monolithic 6', color='tab:purple', linewidth=3)
# plt.plot(m7_major, m7_opt, label='Monolithic 7', color='tab:brown', linewidth=3)
# plt.plot(m8_major, m8_opt, label='Monolithic 8', color='tab:gray', linewidth=3)
# plt.plot(m9_major, m9_opt, label='Monolithic 9', color='tab:pink', linewidth=3)
# plt.plot(m10_major, m10_opt, label='Monolithic 10', color='tab:cyan', linewidth=3)

plt.loglog(obj2, linewidth=3)
# plt.plot(obj3, linewidth=3)
# plt.plot(obj4, linewidth=3)
# plt.plot(obj5, linewidth=3)
# plt.plot(obj6, linewidth=3)
# plt.plot(obj7, linewidth=3)
# plt.plot(obj8, linewidth=3)
# plt.plot(obj9, linewidth=3)
# plt.plot(obj10, linewidth=3)

plt.xlabel('Iteration')
# plt.ylabel('Optimality')
plt.ylabel('Objective function value')
plt.legend(fontsize=8)
plt.yscale('log')
# plt.grid(color='lavender', alpha=0.5)
# plt.ylim(1e1, 1e2)
# plt.savefig('convergence.png', dpi=300, transparent=True, bbox_inches='tight')
plt.show()




exit()


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