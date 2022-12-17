import pathlib

import apts
import cppcolormap as cm
import GooseMPL as gplt
import matplotlib.pyplot as plt
import numpy as np


plt.style.use(["goose", "goose-latex", "goose-huge"])

root = pathlib.Path(__file__).parent
basename = pathlib.Path(__file__).stem

fig, ax = gplt.subplots()

inset = ax.inset_axes([0.25, 0.25, 0.7, 0.7])

cmap = cm.jet(10)
width = np.linspace(0, 7, cmap.shape[0] + 1)[1:]

for w, c in zip(width, cmap):

    particle = apts.Quadratic(w=w)
    v0 = np.logspace(np.log10(particle.vc), 1, 1000)
    vexit = np.NaN * np.ones_like(v0)

    for j in range(v0.size):

        particle.v0 = v0[j]

        if particle.exits:
            vexit[j] = particle.v(particle.tau_exit)

    inset.plot(v0, (v0 - vexit), rasterized=True, c=c)
    inset.plot([particle.vc], [particle.vc], c=c, marker="o")

    ax.plot(
        (v0 - particle.vc) / particle.vc,
        (v0 - vexit) / particle.vc - 2 * particle.w * particle.lam / particle.vc,
        rasterized=True,
        c=c,
    )

    ax.plot(
        (v0 - particle.vc)[0] / particle.vc,
        (v0 - vexit)[0] / particle.vc - 2 * particle.w * particle.lam / particle.vc,
        rasterized=True,
        c=c,
    )

inset.fill_between([0, particle.vc], [0, particle.vc], [particle.vc, particle.vc], color="k", alpha=0.2)
inset.plot([0, particle.vc], [0, particle.vc], c="k", zorder=0)

ax.set_xlim([0, 30])
ax.set_ylim([0, 1])

gplt.xticks(keep=slice(0, None, 2), axis=ax)
gplt.yticks(keep=[0, -1], axis=ax)

ax.xaxis.set_label_coords(0.5, -0.05)
ax.yaxis.set_label_coords(-0.05, 0.5)

ax.set_xlabel(r"$v_0 / v_c$")
ax.set_ylabel(r"$(v_0 - v_\mathrm{exit} - 2 w \lambda) / v_c$")

inset.annotate(
    r"$w$",
    (2, 0),
    xytext=(5, 1),
    arrowprops=dict(arrowstyle="<-", color="k"),
    color="k",
)

inset.set_xlim([0, 10])
inset.set_ylim([0, 1.3])

inset.set_yticks(np.linspace(0, 1.2, 7))
gplt.yticks(keep=[0, -1], axis=inset)
gplt.xticks(keep=[0, -1], axis=inset)

inset.set_xlabel(r"$v_0$")
inset.set_ylabel(r"$v_0 - v_\mathrm{exit}$")

inset.xaxis.set_label_coords(0.5, -0.05)
inset.yaxis.set_label_coords(-0.05, 0.5)

fig.savefig(root / (basename + ".pdf"), bbox_inches="tight")
fig.savefig(root / (basename + ".png"), bbox_inches="tight")
plt.close(fig)
