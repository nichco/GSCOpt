import matplotlib.pyplot as plt
import numpy as np


time = np.linspace(0, 1, 300)

k = 10  # steepness
t0 = 0.5  # midpoint
knowledge = 1 / (1 + np.exp(-k * (time - t0)))
knowledge = 100 * knowledge  # scale to 0–100%



freedom = np.exp(-5 * time)

# Normalize to 100%
knowledge = 100 * knowledge / np.max(knowledge)
freedom = 100 * freedom / np.max(freedom)


# plt.figure(figsize=(10, 6))
plt.plot(time, knowledge, label='KNOWLEDGE ABOUT\nTHE OBJECT OF DESIGN', linewidth=3)
# plt.plot(time, freedom, label='DESIGN\nFREEDOM', linewidth=3)
plt.plot(time, freedom, label='DESIGN FREEDOM', linewidth=3)


plt.fill_between(time, knowledge, 0, alpha=0.2, color='tab:blue')
plt.fill_between(time, freedom, 0, alpha=0.2, color='tab:orange')


plt.title('PARADOX OF ENGINEERING DESIGN', fontsize=14, fontweight='bold', ha='center')
plt.xlabel('TIME INTO DESIGN PROCESS')
plt.ylabel('%')

plt.xlim(0, 1)
plt.ylim(0, 100)


plt.legend(fontsize=14)


plt.grid(False)
plt.tight_layout()
plt.show()
