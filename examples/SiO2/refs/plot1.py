import matplotlib.pyplot as plt
import numpy as np

#plt.rcParams["font.family"] = "serif"
#plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams.update({'font.size': 14}) # Sets a global font size of 12

# Load data (Converting your loadtxt block to be more compact, 
# but keeping your logic intact)
data1 = np.loadtxt('dft_ipa.dat')
data2 = np.loadtxt('gw_ipa.dat')
data3 = np.loadtxt('dft_lf.dat')
data4 = np.loadtxt('gw_lf.dat')
data5 = np.loadtxt('bse_gw_vmn.dat')
data6 = np.loadtxt('bse_dft_vmn.dat')
data7 = np.loadtxt('bse_tuned.dat')
data8 = np.loadtxt('bse_acc.dat')
data9 = np.loadtxt('exp.dat')

# Assign variables (Preserving your exact math logic)
x1, y1 = data1[:,0], data1[:,9]
x2, y2 = data2[:,0], data2[:,9]
x3, y3 = data3[:,0], data3[:,9]*2 # Preserved the *2 scaling
x4, y4 = data4[:,0], data4[:,9]*2 # Preserved the *2 scaling
x5, y5 = data5[:,0], data5[:,9]
x6, y6 = data6[:,0], data6[:,9]
x7, y7 = data7[:,0], data7[:,9]
x8, y8 = data8[:,0], data8[:,9]
x9, y9 = data9[:,0], data9[:,1]

# Set up figure with a standard academic aspect ratio
fig, ax = plt.subplots(figsize=(6, 4.5))

# --- Group 1: DFT Based (Blue Tones) ---
ax.plot(x1, y1, label='DFT-IPA', color='tab:purple', linestyle=':', linewidth=1.5)
ax.plot(x3, y3, label='DFT-LFC',  color='tab:purple', linestyle='--', linewidth=1.5)

# --- Group 2: GW Based (Red Tones) ---
ax.plot(x2, y2, label='$GW$-IPA', color='tab:olive', linestyle=':', linewidth=2)
ax.plot(x4, y4, label='$GW$-LFC',  color='tab:olive', linestyle='--', linewidth=2)

# --- Group 3: BSE (Highest Accuracy - Solid/Distinct Lines) ---
# Using distinct colors for the "Best" theories to make them stand out
ax.plot(x5, y5, label=r'$GW$-BSE ($\mathbf{v}_{GW}$)',      color='tab:green',  linestyle='--',  linewidth=2.5, alpha=1)
ax.plot(x6, y6, label=r'$GW$-BSE ($\mathbf{v}_\text{DFT}$)',     color='tab:blue',  linestyle=':', linewidth=2.5, alpha=1)
ax.plot(x7, y7, label=r'$GW$-BSE ($\mathbf{v}_\text{opt}$)', color='tab:red', linestyle='-.',  linewidth=2.5, alpha=1)
# ax.plot(x8, y8, label='GW-BSE-SOS', color='tab:brown')

# --- Group 4: Experiment (Black, on top) ---
ax.scatter(x9, y9, label='Exp.', color='black', marker='D', s=60, linewidth=0.5, 
           facecolors='none', zorder=10) # zorder=10 ensures it sits on top of lines

# Labels and Limits
xl = r'$\hbar \omega \, \mathrm{(eV)}$'
yl = r'$\rho \, \mathrm{(deg/mm)}$'

ax.set_xlabel(xl)
ax.set_ylabel(yl)
ax.set_xlim(0, 6)
ax.set_ylim(-100, 400)

# Improved Legend
# 'frameon=False' looks cleaner for publications
# 'ncol=2' splits the long list into two columns so it doesn't cover data vertically
ax.legend(loc='upper left', ncol=2, fontsize=12, frameon=False)

plt.tight_layout()
plt.savefig("oa-quartz.eps", bbox_inches="tight")
plt.show()
