import pathlib

import apts
import GooseMPL as gplt
import matplotlib.pyplot as plt
import numpy as np


plt.style.use(["goose", "goose-latex", "goose-huge"])

root = pathlib.Path(__file__).parent
basename = pathlib.Path(__file__).stem

fig, axr = gplt.subplots()
axv = axr.twinx()

axr.set_xlabel(r"$\lambda t$")
axr.set_ylabel("$r / w$")
axv.set_ylabel("$v / Q$", color="r")
axv.tick_params(axis="y", labelcolor="r")

particle = apts.Quadratic(v0=3)
vc = particle.vc
tau0 = 0
r0 = 0

ret_tau = []
ret_r = []
ret_v = []


while True:

    if not particle.exits:
        break

    tau = np.linspace(0, particle.tau_exit, 1000)
    r = particle.r(tau)
    r += r0 - r[0]

    ret_tau += list(tau0 + tau)
    ret_r += list(r)
    ret_v += list(particle.v(tau))

    r0 = r[-1]
    tau0 += tau[-1]

    particle.v0 = particle.v(particle.tau_exit)

    if particle.v0 < vc:
        tauc = tau0

taulim = 3 / particle.lam
tau = np.linspace(0, taulim - tau0, 1000)
r = particle.r(tau)
r += r0 - r[0]

ret_tau += list(tau0 + tau)
ret_r += list(r)
ret_v += list(particle.v(tau))

tau = np.array(ret_tau) * particle.lam
r = np.array(ret_r) / particle.w
v = np.array(ret_v) / particle.Q
vc = vc / particle.Q
tauc = tauc * particle.lam

axr.plot(tau, r, c="k")
axv.plot(tau, v, c="r")
axv.axhline(vc, c="r", ls="--", lw=1)

i = np.argmin(np.abs(tau - 0.6))

axr.annotate(
    r"$r$", (tau[i], r[i]), xytext=(tau[i] * 1.2, r[i] * 0.95), arrowprops=dict(arrowstyle="->")
)

axv.annotate(
    r"$v$",
    (tau[i], v[i]),
    xytext=(tau[i] * 1.2, v[i] * 1.3),
    arrowprops=dict(arrowstyle="->", color="r"),
    color="r",
)

i = np.argmin(np.abs(tau - 3))

axv.annotate(
    r"$v_c$",
    (tau[i], vc),
    xytext=(tau[i] * 0.92, vc * 1.5),
    arrowprops=dict(arrowstyle="->", color="r"),
    color="r",
)

axr.set_xlim([0, tau[-1]])
axr.set_ylim([0, axr.get_ylim()[1]])

axr.fill_between(
    [tauc, axr.get_xlim()[1]],
    [axr.get_ylim()[0], axr.get_ylim()[0]],
    [axr.get_ylim()[1], axr.get_ylim()[1]],
    color="k",
    alpha=0.1,
)

axv.set_ylim([-1, 6])

fig.savefig(root / (basename + ".pdf"), bbox_inches="tight")
fig.savefig(root / (basename + ".png"), bbox_inches="tight")
plt.close(fig)
