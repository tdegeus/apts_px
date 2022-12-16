import matplotlib.pyplot as plt
import numpy as np
import apts
import GooseMPL as gplt
import pathlib


plt.style.use(["goose", "goose-latex", "goose-autolayout"])

root = pathlib.Path(__file__).parent
basename = pathlib.Path(__file__).stem

fig, axr = gplt.subplots()
axv = axr.twinx()

particle = apts.Quadratic(v0=5)
tau0 = 0
r0 = 0

while True:

    if not particle.exits:
        break

    tau = np.linspace(0, particle.tau_exit, 1000)
    r = particle.r(tau)
    r += (r0 - r[0])
    axr.plot(tau0 + tau, r, c="k")
    axv.plot(tau0 + tau, particle.v(tau), c="r")

    r0 = r[-1]
    tau0 += tau[-1]

    particle.v0 = particle.v(particle.tau_exit)

tau = np.linspace(0, 30, 1000)
r = particle.r(tau)
r += (r0 - r[0])
axr.plot(tau0 + tau, r, c="k")
axv.plot(tau0 + tau, particle.v(tau), c="r")


# axr.plot(tau, particle.r(tau))

# axr.set_ylabel('$r$')
# axr.set_xlabel(r'$\tau$')

# axr.axhline(-0.5, c="k", ls="--", lw=1)
# axr.axhline(0.5, c="k", ls="--", lw=1)
# axr.axvline(particle.tau_exit, c="k", ls="--", lw=1)

# axr.set_xlim([0, taumax])
# axr.set_ylim([-0.75, 0.75])



# axv.set_ylabel('$v$', color="r")

# axv.tick_params(axis ='y', labelcolor='r')

# axv.set_ylim([-0.75, 0.75])

# fig.savefig(root / (basename + ".pdf"))
# fig.savefig(root / (basename + ".png"))
# plt.close(fig)

plt.show()
