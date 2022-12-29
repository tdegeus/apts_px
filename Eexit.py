import pathlib

import apts
import cppcolormap as cm
import enstat
import GooseMPL as gplt
import h5py
import matplotlib.pyplot as plt
import numpy as np

plt.style.use(["goose", "goose-latex", "goose-huge"])

root = pathlib.Path(__file__).parent
basename = pathlib.Path(__file__).stem


def plot_triagle(x0, y0, w, h, ax, label):

    ax.plot([x0, x0 + w], [y0, y0], c="k", lw=1, transform=ax.transAxes)
    ax.plot([x0 + w, x0 + w], [y0, y0 + h], c="k", lw=1, transform=ax.transAxes)
    ax.plot([x0, x0 + w], [y0, y0 + h], c="k", lw=1, transform=ax.transAxes)
    ax.text(x0 + 0.5 * w, y0 - 0.1 * h, r"$1$", ha="center", va="top", transform=ax.transAxes)
    ax.text(x0 + w + 0.1 * w, y0 + 0.5 * h, label, ha="left", va="center", transform=ax.transAxes)


fig, ax = gplt.subplots()

inset = ax.inset_axes([0.1, 0.58, 0.4, 0.39])
inset2 = ax.inset_axes([0.56, 0.07, 0.4, 0.39])

ax.set_xscale("log")
ax.set_yscale("log")

inset.set_xscale("log")
inset.set_yscale("log")

inset2.set_xscale("log")
inset2.set_yscale("log")

inset.set_xlim([1e-2, 1e3])
inset.set_ylim([1e-2, 1e2])

cmap = cm.jet(10)
width = np.linspace(0, 10, cmap.shape[0] + 1)[1:]

for w, c in zip(width, cmap):

    particle = apts.Quadratic(w=w)
    v0 = np.logspace(np.log10(particle.vc), 2, 1000)
    vexit = np.NaN * np.ones_like(v0)

    for j in range(v0.size):

        particle.v0 = v0[j]

        if particle.exits:
            vexit[j] = particle.v(particle.tau_exit)

    Ec = 0.5 * particle.vc**2
    E0 = 0.5 * v0**2
    Eexit = 0.5 * vexit**2

    ax.plot(np.sqrt(E0) / w, (E0 - Eexit) / w**2, c=c)
    inset.plot(E0, E0 - Eexit, c=c)

    ax.plot([np.sqrt(Ec) / w], [Ec / w**2], c=c, marker="o")
    inset.plot([Ec], [Ec], c=c, marker="o")

inset.fill_between([0, 100], [0, 100], [100, 100], color="k", alpha=0.2)
inset.plot([0, 100], [0, 100], c="k", zorder=0)

inset.annotate(
    r"$w$",
    (1e2, 5e1),
    xytext=(1e2, 1e-1),
    arrowprops=dict(arrowstyle="->", color="k"),
    color="k",
)

gplt.plot_powerlaw(0.5, 0, 0, 1, axis=inset, color="k", ls="--")

ax.xaxis.set_label_coords(0.5, -0.05)
ax.yaxis.set_label_coords(-0.05, 0.5)

gplt.log_yticks(keep=[0, -1], axis=ax)
gplt.log_xticks(keep=[0, -1], axis=ax)

inset.xaxis.set_label_coords(0.5, -0.05)
inset.yaxis.set_label_coords(-0.05, 0.5)

gplt.log_yticks(keep=[0, -1], axis=inset)
gplt.log_xticks(keep=[0, -1], axis=inset)

gplt.plot_powerlaw(1, 0.067, 0, 1, axis=ax, color="k", ls="--")

ax.set_xlabel(r"$\sqrt{E_0} / w$")
ax.set_ylabel(r"$(E_0 - E_\mathrm{exit}) / w^2$")

inset.set_xlabel(r"$E_0$")
inset.set_ylabel(r"$E_0 - E_\mathrm{exit}$")

plot_triagle(0.65, 0.6, 0.1, 0.11, ax, r"$1$")
plot_triagle(0.39, 0.2, 0.2, 0.12, inset, r"$\tfrac{1}{2}$")

# ---

ensemble = {
    "distro=power_k=1_fext=0,1.h5": (0, cm.Purples(6)[3], "."),
    "distro=power_k=2_fext=0,1.h5": (1, cm.Purples(6)[4], "+"),
    "distro=power_k=3_fext=0,1.h5": (2, cm.Purples(6)[5], "x"),
    "distro=weibull_k=1_fext=0,1.h5": (0, cm.Oranges(6)[3], "."),
    "distro=weibull_k=2_fext=0,1.h5": (1, cm.Oranges(6)[4], "+"),
    "distro=weibull_k=3_fext=0,1.h5": (2, cm.Oranges(6)[5], "x"),
    "distro=random_fext=0,1.h5": (0, "g", "."),
    "distro=power_k=1.h5": (0, cm.Blues(6)[3], "."),
    "distro=power_k=2.h5": (1, cm.Blues(6)[4], "+"),
    "distro=power_k=3.h5": (2, cm.Blues(6)[5], "x"),
    "distro=weibull_k=1.h5": (0, cm.Reds(6)[3], "."),
    "distro=weibull_k=2.h5": (1, cm.Reds(6)[4], "+"),
    "distro=weibull_k=3.h5": (2, cm.Reds(6)[5], "x"),
    "distro=random.h5": (0, "k", "."),
}

for filename, (_, color, marker) in ensemble.items():

    with h5py.File(root / "data" / filename) as file:
        hist = (
            enstat.histogram(
                bin_edges=file["/Estop/narrow/bin_edges"][...],
                count=file["/Estop/narrow/count"][...],
            )
            .strip()
            .squash(6)
        )

    x = hist.x
    y = hist.p
    y /= y[0]

    inset2.plot(x, y, c=color, rasterized=True)

inset2.set_xlabel(r"$E_0$")
inset2.set_ylabel(r"$P(E_0) / c_e$")

inset2.set_xlim([1e-5, 1e-1])
inset2.set_ylim([1e-2, 2e0])

inset2.axhline(1, c="b", ls="-")

gplt.log_xticks(keep=[0, -1], axis=inset2)
gplt.log_yticks(keep=[0, -1], axis=inset2)

inset2.xaxis.set_label_coords(0.5, -0.05)
inset2.yaxis.set_label_coords(-0.05, 0.5)

fig.savefig(root / (basename + ".pdf"), bbox_inches="tight")
fig.savefig(root / (basename + ".png"), bbox_inches="tight")
plt.close(fig)
