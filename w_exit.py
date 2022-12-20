import pathlib

import apts
import GooseMPL as gplt
import matplotlib.pyplot as plt
import prrng

plt.style.use(["goose", "goose-latex", "goose-autolayout"])

root = pathlib.Path(__file__).parent
basename = pathlib.Path(__file__).stem

k = 2
v0 = prrng.pcg32().normal([int(3e5)], 10, 1)
particle = apts.Quadratic()
wstop = apts.throw_particles_Quadratic(v0, particle, prrng.distribution.power, [k + 1, 1e-10])


fig, ax = plt.subplots()

w = prrng.pcg32().power([int(3e5)], k + 1)
P, x = gplt.cdf(w)
ax.plot(x, P, rasterized=True, marker=".", c="r", label=r"$\phi(w)$")
ax.plot(x, x * P, rasterized=True, marker=".", c="b", label=r"$w \phi(w)$")

P, x = gplt.cdf(wstop)
ax.plot(x, P, rasterized=True, marker=".", c="k", label=r"$\phi_\mathrm{stop}(w)$")

ax.legend()

ax.set_xlabel(r"$w$")
ax.set_ylabel(r"$\phi(w)$")

ax.set_xscale("log")
ax.set_yscale("log")

fig.savefig(root / (basename + ".pdf"), bbox_inches="tight")
fig.savefig(root / (basename + ".png"), bbox_inches="tight")
plt.close(fig)
