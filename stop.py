import pathlib

import cppcolormap as cm
import enstat
import GooseMPL as gplt
import h5py
import matplotlib.pyplot as plt
import numpy as np

plt.style.use(["goose", "goose-latex", "goose-autolayout"])

root = pathlib.Path(__file__).parent
basename = pathlib.Path(__file__).stem

colors = {
    "distro=random.h5": "k",
    "distro=weibull_k=1.h5": cm.Reds(6)[3],
    "distro=weibull_k=2.h5": cm.Reds(6)[4],
    "distro=weibull_k=3.h5": cm.Reds(6)[5],
    "distro=power_k=1.h5": cm.Greens(6)[3],
    "distro=power_k=2.h5": cm.Greens(6)[4],
    "distro=power_k=3.h5": cm.Greens(6)[5],
    "distro=random_fext=0,1.h5": "k",
    "distro=weibull_k=1_fext=0,1.h5": cm.Reds(6)[3],
    "distro=weibull_k=2_fext=0,1.h5": cm.Reds(6)[4],
    "distro=weibull_k=3_fext=0,1.h5": cm.Reds(6)[5],
    "distro=power_k=1_fext=0,1.h5": cm.Greens(6)[3],
    "distro=power_k=2_fext=0,1.h5": cm.Greens(6)[4],
    "distro=power_k=3_fext=0,1.h5": cm.Greens(6)[5],
}

disorder = {
    "distro=random.h5": 0,
    "distro=weibull_k=1.h5": 0,
    "distro=weibull_k=2.h5": 1,
    "distro=weibull_k=3.h5": 2,
    "distro=power_k=1.h5": 0,
    "distro=power_k=2.h5": 1,
    "distro=power_k=3.h5": 2,
    "distro=random_fext=0,1.h5": 0,
    "distro=weibull_k=1_fext=0,1.h5": 0,
    "distro=weibull_k=2_fext=0,1.h5": 1,
    "distro=weibull_k=3_fext=0,1.h5": 2,
    "distro=power_k=1_fext=0,1.h5": 0,
    "distro=power_k=2_fext=0,1.h5": 1,
    "distro=power_k=3_fext=0,1.h5": 2,
}

# -----

fig, ax = plt.subplots()

inset = ax.inset_axes([0.08, 0.56, 0.44, 0.4], facecolor="w")

ax.set_xscale("log")
ax.set_yscale("log")

inset.set_xscale("log")
inset.set_yscale("log")

ax.set_xlabel(r"$w_\mathrm{stop}$")
ax.set_ylabel(r"$P(w_\mathrm{stop}) / c_w$")

inset.set_xlabel(r"$v$")
inset.set_ylabel(r"$P(v) / c_v$")

for name, c in colors.items():

    with h5py.File(root / "data" / name) as file:
        hist = enstat.histogram(
            bin_edges=file["/vstop/bin_edges"][...],
            count=file["/vstop/count"][...],
        ).strip()

    p = hist.p
    x = hist.x
    keep = x < 0.7 * x[np.argmax(p)]
    keep[0] = False

    _, _, param = gplt.fit_powerlaw(x[keep], p[keep], exponent=1)
    prefactor = param["prefactor"]

    inset.plot(x, p / prefactor, c=c, marker=".", ls="none", rasterized=True)

inset.set_xlim([2e-4, 3e-1])
inset.set_ylim([3e-4, 2e-1])

x = np.logspace(np.log10(inset.get_xlim()[0]), np.log10(inset.get_xlim()[1]), 1000)
y = x**1

inset.plot(x, y, c="b")

gplt.log_xticks(keep=[0, -1], axis=inset)
gplt.log_yticks(keep=[0, -1], axis=inset)

inset.xaxis.set_label_coords(0.5, -0.05)
inset.yaxis.set_label_coords(-0.05, 0.5)

# -------

for name, c in colors.items():

    with h5py.File(root / "data" / name) as file:
        hist = (
            enstat.histogram(
                bin_edges=file["/wstop/bin_edges"][...],
                count=file["/wstop/count"][...],
            )
            .strip()
            .squash(4)
        )

    p = hist.p
    x = hist.x
    keep = x < 0.4 * x[np.argmax(p)]
    keep[0] = False

    _, _, param = gplt.fit_powerlaw(x[keep], p[keep], exponent=disorder[name] + 2)
    prefactor = param["prefactor"]

    ax.plot(x, p / prefactor, c=c, marker=".", ls="none", rasterized=True)

ax.set_xlim([2e-2, 1e0])
ax.set_ylim([1e-5, 2e0])

for k, c in zip([0, 1, 2], cm.Blues(6)[3:]):
    x = np.logspace(np.log10(ax.get_xlim()[0]), np.log10(ax.get_xlim()[1]), 1000)
    y = x ** (k + 2)
    ax.plot(x, y, c="b")

gplt.log_xticks(keep=[0, -1], axis=ax)
gplt.log_yticks(keep=[0, -1], axis=ax)

ax.xaxis.set_label_coords(0.5, -0.05)
ax.yaxis.set_label_coords(-0.05, 0.5)

fig.savefig(root / (basename + ".pdf"), bbox_inches="tight")
fig.savefig(root / (basename + ".png"), bbox_inches="tight")
plt.close(fig)
