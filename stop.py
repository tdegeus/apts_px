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


def plot_triagle(x0, y0, w, h, ax, label):

    ax.plot([x0, x0 + w], [y0, y0], c="b", lw=1, transform=ax.transAxes)
    ax.plot([x0 + w, x0 + w], [y0, y0 + h], c="b", lw=1, transform=ax.transAxes)
    ax.plot([x0, x0 + w], [y0, y0 + h], c="b", lw=1, transform=ax.transAxes)
    ax.text(
        x0 + 0.5 * w, y0 - 0.1 * h, r"$1$", c="b", ha="center", va="top", transform=ax.transAxes
    )
    ax.text(
        x0 + w + 0.1 * w, y0 + 0.5 * h, label, c="b", ha="left", va="center", transform=ax.transAxes
    )


def plot_inverse_triagle(x0, y0, w, h, ax, label):

    ax.plot([x0, x0 + w], [y0 + h, y0 + h], c="b", lw=1, transform=ax.transAxes)
    ax.plot([x0, x0], [y0, y0 + h], c="b", lw=1, transform=ax.transAxes)
    ax.plot([x0, x0 + w], [y0, y0 + h], c="b", lw=1, transform=ax.transAxes)
    ax.text(
        x0 + 0.5 * w,
        y0 + h + 0.1 * h,
        r"$1$",
        c="b",
        ha="center",
        va="bottom",
        transform=ax.transAxes,
    )
    ax.text(
        x0 - 0.1 * w, y0 + 0.5 * h, label, c="b", ha="right", va="center", transform=ax.transAxes
    )


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

for name, (exponent, color, marker) in ensemble.items():

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

    inset.plot(x, p / prefactor, c=color, marker=marker, ls="none", rasterized=True)

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

for name, (exponent, color, marker) in ensemble.items():

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

    _, _, param = gplt.fit_powerlaw(x[keep], p[keep], exponent=exponent + 2)
    prefactor = param["prefactor"]

    ax.plot(x, p / prefactor, c=color, marker=marker, ls="none", rasterized=True)

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

plot_triagle(0.47, 0.4, 0.2, 0.225, inset, "$1$")

plot_triagle(0.25, 0.44, 0.1, 0.13 / 2, ax, "$2$")
plot_inverse_triagle(0.1, 0.14, 0.1, 0.13 / 4 * 3, ax, "$3$")
plot_triagle(0.53, 0.3, 0.1, 0.13, ax, "$4$")

x0 = 0.85
y0 = 0.05
dx = 0.05
dy = 0.05

r = 6
ax.text(x0 - 0.5 * dx, y0 + r * dy, "$k = $", ha="right", va="center", transform=ax.transAxes)
ax.text(x0, y0 + r * dy, "$0$", ha="center", va="center", transform=ax.transAxes)
ax.text(x0 + dx, y0 + r * dy, "$1$", ha="center", va="center", transform=ax.transAxes)
ax.text(x0 + 2 * dx, y0 + r * dy, "$2$", ha="center", va="center", transform=ax.transAxes)

ax.plot(
    [x0 - 3 * dx, x0 + 2.5 * dx],
    [y0 + (r - 0.5) * dy, y0 + (r - 0.5) * dy],
    c="k",
    lw=1,
    transform=ax.transAxes,
)

r = 5
ax.text(
    x0 - 0.5 * dx,
    y0 + r * dy - 0.5 * dy,
    "weibull",
    ha="right",
    va="center",
    transform=ax.transAxes,
)
ax.plot(x0, y0 + r * dy, c=cm.Reds(6)[3], marker=".", ls="none", transform=ax.transAxes)
ax.plot(x0 + dx, y0 + r * dy, c=cm.Reds(6)[4], marker="+", ls="none", transform=ax.transAxes)
ax.plot(x0 + 2 * dx, y0 + r * dy, c=cm.Reds(6)[5], marker="x", ls="none", transform=ax.transAxes)

r = 4
ax.plot(x0, y0 + r * dy, c=cm.Orange(6)[3], marker=".", ls="none", transform=ax.transAxes)
ax.plot(x0 + dx, y0 + r * dy, c=cm.Orange(6)[4], marker="+", ls="none", transform=ax.transAxes)
ax.plot(x0 + 2 * dx, y0 + r * dy, c=cm.Orange(6)[5], marker="x", ls="none", transform=ax.transAxes)

ax.plot(
    [x0 - 3 * dx, x0 + 2.5 * dx],
    [y0 + (r - 0.5) * dy, y0 + (r - 0.5) * dy],
    c="k",
    lw=1,
    transform=ax.transAxes,
)

r = 3
ax.text(
    x0 - 0.5 * dx, y0 + r * dy - 0.5 * dy, "power", ha="right", va="center", transform=ax.transAxes
)
ax.plot(x0, y0 + r * dy, c=cm.Blues(6)[3], marker=".", ls="none", transform=ax.transAxes)
ax.plot(x0 + dx, y0 + r * dy, c=cm.Blues(6)[4], marker="+", ls="none", transform=ax.transAxes)
ax.plot(x0 + 2 * dx, y0 + r * dy, c=cm.Blues(6)[5], marker="x", ls="none", transform=ax.transAxes)

r = 2
ax.plot(x0, y0 + r * dy, c=cm.Purples(6)[3], marker=".", ls="none", transform=ax.transAxes)
ax.plot(x0 + dx, y0 + r * dy, c=cm.Purples(6)[4], marker="+", ls="none", transform=ax.transAxes)
ax.plot(x0 + 2 * dx, y0 + r * dy, c=cm.Purples(6)[5], marker="x", ls="none", transform=ax.transAxes)

ax.plot(
    [x0 - 3 * dx, x0 + 2.5 * dx],
    [y0 + (r - 0.5) * dy, y0 + (r - 0.5) * dy],
    c="k",
    lw=1,
    transform=ax.transAxes,
)

r = 1
ax.text(
    x0 - 0.5 * dx,
    y0 + r * dy - 0.5 * dy,
    "uniform",
    ha="right",
    va="center",
    transform=ax.transAxes,
)
ax.plot(x0, y0 + r * dy, c="k", marker=".", ls="none", transform=ax.transAxes)

r = 0
ax.plot(x0, y0 + r * dy, c="g", marker=".", ls="none", transform=ax.transAxes)

ax.plot(
    [x0 - 3 * dx, x0 + 2.5 * dx],
    [y0 + (r - 0.5) * dy, y0 + (r - 0.5) * dy],
    c="k",
    lw=1,
    transform=ax.transAxes,
)

fig.savefig(root / (basename + ".pdf"), bbox_inches="tight")
fig.savefig(root / (basename + ".png"), bbox_inches="tight")
plt.close(fig)
