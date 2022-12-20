import pathlib

import apts
import GooseMPL as gplt
import matplotlib.pyplot as plt
import numpy as np
import prrng

plt.style.use(["goose", "goose-latex", "goose-huge"])

root = pathlib.Path(__file__).parent
basename = pathlib.Path(__file__).stem

fig, ax = gplt.subplots()

ax.set_xlabel("$v_0$")
ax.set_ylabel(r"$P(v_0)$")

ax.set_xscale("log")
ax.set_yscale("log")

inset = ax.inset_axes([0.3, 0.16, 0.6, 0.6])

inset.set_xlabel(r"$w$")
inset.set_ylabel(r"$v_c$")

ret = []

for seed in range(1000):

    w = prrng.pcg32(seed).gamma([3000], 2)
    f = prrng.pcg32(seed).normal([3000], 0.1, 0.01)
    vexit = [prrng.pcg32(seed).normal([], 200, 1)]
    particle = apts.Quadratic(v0=vexit[-1], w=w[0], f=f[0])

    for i in range(1, w.size):

        if not particle.exits:
            break

        vexit.append(float(particle.v(particle.tau_exit)))
        particle = apts.Quadratic(v0=vexit[i], w=w[i], f=f[i])

    ret += vexit

v = np.array(ret)
bin_edges = gplt.histogram_bin_edges(v, bins=300, mode="log")
P, x = gplt.histogram(v, bins=bin_edges, density=True, return_edges=False)

ax.plot(x, P, rasterized=True, marker=".")

gplt.plot_powerlaw(1, 0.1, 0, 1, axis=ax, c="b")

# ax.set_xlim([1e-2, 1.1e2])
# ax.set_ylim([1e-3, 2e0])

# gplt.log_xticks(keep=[0, -1], axis=ax)
# gplt.log_yticks(keep=[0, -1], axis=ax)

# ax.xaxis.set_label_coords(0.5, -0.05)
# ax.yaxis.set_label_coords(-0.05, 0.5)

# ---

w = np.linspace(0, 10, 1000)
vc = np.array([apts.Quadratic(w=i).vc for i in w])

inset.plot(w, vc)

inset.set_xlim([0, 10])
inset.set_ylim([0, 3])

gplt.xticks(keep=[0, -1], axis=inset)
gplt.yticks(keep=[0, -1], axis=inset)

inset.xaxis.set_label_coords(0.5, -0.05)
inset.yaxis.set_label_coords(-0.05, 0.5)

fig.savefig(root / (basename + ".pdf"), bbox_inches="tight")
fig.savefig(root / (basename + ".png"), bbox_inches="tight")
plt.close(fig)
