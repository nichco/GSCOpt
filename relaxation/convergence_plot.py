import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

filename2 = 'relaxation/convergence/monolithic_2.out'
filename3 = 'relaxation/convergence/monolithic_3.out'
filename4 = 'relaxation/convergence/monolithic_4.out'
filename5 = 'relaxation/convergence/monolithic_5.out'
filename6 = 'relaxation/convergence/monolithic_6.out'

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

# Read the rest of the data into a DataFrame
m2 = pd.read_csv(filename2, delim_whitespace=True, skiprows=1, names=headers2)
m3 = pd.read_csv(filename3, delim_whitespace=True, skiprows=1, names=headers3)
m4 = pd.read_csv(filename4, delim_whitespace=True, skiprows=1, names=headers4)
m5 = pd.read_csv(filename5, delim_whitespace=True, skiprows=1, names=headers5)
m6 = pd.read_csv(filename6, delim_whitespace=True, skiprows=1, names=headers6)

# Print the DataFrame to verify
# print(m2)

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

plt.plot(m2_major, m2_obj, label='Monolithic 2 copies', color='tab:blue')
plt.plot(m3_major, m3_obj, label='Monolithic 3 copies', color='tab:orange')
plt.plot(m4_major, m4_obj, label='Monolithic 4 copies', color='tab:green')
plt.plot(m5_major, m5_obj, label='Monolithic 5 copies', color='tab:red')
plt.plot(m6_major, m6_obj, label='Monolithic 6 copies', color='tab:purple')
# plt.plot(m2_major, m2_opt, label='Monolithic 2 copies', color='tab:blue')
# plt.plot(m3_major, m3_opt, label='Monolithic 3 copies', color='tab:orange')
# plt.plot(m4_major, m4_opt, label='Monolithic 4 copies', color='tab:green')
# plt.plot(m5_major, m5_opt, label='Monolithic 5 copies', color='tab:red')
# plt.plot(m6_major, m6_opt, label='Monolithic 6 copies', color='tab:purple')
plt.xlabel('Iteration')
plt.ylabel('Optimality')
plt.legend()
plt.yscale('log')
plt.ylim(1e1, 1e2)
plt.show()