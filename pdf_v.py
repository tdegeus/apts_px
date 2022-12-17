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
ax.set_ylabel(r"$\Phi(v_0)$")

ax.set_xscale("log")
ax.set_yscale("log")

inset = ax.inset_axes([0.12, 0.12, 0.7, 0.7])

inset.set_xlabel(r"$v_0 - v_\mathrm{exit}$")
inset.set_ylabel(r"$\Phi(v_0 - v_\mathrm{exit})$")

inset.set_xscale("log")
inset.set_yscale("log")

ret = []

for seed in range(200):

    v0 = 100
    w = prrng.pcg32(seed).normal([1000], 1, 0.5)
    vexit = [v0]
    particle = apts.Quadratic(v0=v0, w=w[0])

    for i in range(1, w.size):

        if not particle.exits:
            break

        vexit.append(particle.v(particle.tau_exit))
        particle = apts.Quadratic(v0=vexit[i], w=w[i])

    ret += vexit

v = np.array(ret)
P, x = gplt.ccdf(v)

ax.plot(x, P, rasterized=True, marker=".")

ax.set_xlim([1e-2, 1.1e2])
ax.set_ylim([1e-3, 2e0])

gplt.log_xticks(keep=[0, -1], axis=ax)
gplt.log_yticks(keep=[0, -1], axis=ax)

ax.xaxis.set_label_coords(0.5, -0.05)
ax.yaxis.set_label_coords(-0.05, 0.5)

# ---

w = prrng.pcg32().normal([1000], 1, 0.3)
v = np.logspace(-6, 0, 1000)[::-1]
d = np.NaN * np.ones((w.size, v.size))

for i in range(w.size):

    particle = apts.Quadratic(w=w[i])

    for j in range(v.size):

        particle.v0 = v[j]

        if not particle.exits:
            break

        d[i, j] = v[j] - particle.v(particle.tau_exit)

d = d.ravel()
d = d[~np.isnan(d)]
P, x = gplt.ccdf(d)

inset.plot(x, P, rasterized=True, marker=".")

inset.set_xlim([5e-3, 6e-1])
inset.set_ylim([1e-5, 2e0])

gplt.log_xticks(keep=[0, -1], axis=inset)
gplt.log_yticks(keep=[0, -1], axis=inset)

inset.xaxis.set_label_coords(0.5, -0.05)
inset.yaxis.set_label_coords(-0.05, 0.5)

fig.savefig(root / (basename + ".pdf"), bbox_inches="tight")
fig.savefig(root / (basename + ".png"), bbox_inches="tight")
plt.close(fig)
