import pathlib

import apts
import GooseMPL as gplt
import matplotlib.pyplot as plt
import numpy as np

plt.style.use(["goose", "goose-latex", "goose-autolayout"])

root = pathlib.Path(__file__).parent
basename = pathlib.Path(__file__).stem

fig, axr = gplt.subplots()
axv = axr.twinx()

axr.set_xlabel(r"$\tau$")
axr.set_ylabel("$r$")
axv.set_ylabel("$v$", color="r")
axv.tick_params(axis="y", labelcolor="r")

particle = apts.Quadratic(v0=1, w=2)
taumax = 10
tau = np.linspace(0, taumax, 1000)

axr.set_xlim([0, taumax])
axr.set_ylim([-1.4, 1.4])
axv.set_ylim([-1.4, 1.4])

axr.plot(tau, particle.r(tau))

axr.axhline(-1, c="k", ls="--", lw=1)
axr.axhline(1, c="k", ls="--", lw=1)
axr.axvline(particle.tau_exit, c="k", ls="--", lw=1)

axv.plot(tau, particle.v(tau), c="r")

fig.savefig(root / (basename + ".pdf"))
fig.savefig(root / (basename + ".png"))
plt.close(fig)
