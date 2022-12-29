import argparse

import apts
import enstat
import h5py
import numpy as np
import prrng
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-D", "--distribution", type=str, default="power")
parser.add_argument("-k", type=float, default=1.0)
parser.add_argument("-f", "--fext", type=float)
parser.add_argument("-a", "--append", action="store_true")
parser.add_argument("file", type=str)
args = parser.parse_args()

ngroup = int(5000)
nbatch = int(1e5)

extra = ["prrng=" + prrng.version(), "enstat=" + enstat.version]
deps = sorted(list(set(list(apts.version_dependencies()) + extra)))

with h5py.File(args.file, "a" if args.append else "w") as file:

    if not args.append:

        file["/batch"] = int(0)

        file["/wstop/log/bin_edges"] = np.logspace(-3, 1, int(1e5) + 1)
        file["/vstop/log/bin_edges"] = np.logspace(-4, 1, int(1e5) + 1)
        file["/Estop/log/bin_edges"] = np.logspace(-6, 1, int(1e5) + 1)
        file["/wstop/log/count"] = np.zeros(int(1e5), dtype=np.uint64)
        file["/vstop/log/count"] = np.zeros(int(1e5), dtype=np.uint64)
        file["/Estop/log/count"] = np.zeros(int(1e5), dtype=np.uint64)

        file["/wstop/broad/bin_edges"] = np.linspace(1e-6, 5e1, int(1e5) + 1)
        file["/vstop/broad/bin_edges"] = np.linspace(1e-6, 5e1, int(1e5) + 1)
        file["/Estop/broad/bin_edges"] = np.linspace(1e-6, 5e1, int(1e5) + 1)
        file["/wstop/broad/count"] = np.zeros(int(1e5), dtype=np.uint64)
        file["/vstop/broad/count"] = np.zeros(int(1e5), dtype=np.uint64)
        file["/Estop/broad/count"] = np.zeros(int(1e5), dtype=np.uint64)

        file["/wstop/narrow/bin_edges"] = np.linspace(1e-2, 1e0, int(1e5) + 1)
        file["/vstop/narrow/bin_edges"] = np.linspace(1e-3, 3e-1, int(1e5) + 1)
        file["/Estop/narrow/bin_edges"] = np.linspace(1e-5, 1e-1, int(1e5) + 1)
        file["/wstop/narrow/count"] = np.zeros(int(1e5), dtype=np.uint64)
        file["/vstop/narrow/count"] = np.zeros(int(1e5), dtype=np.uint64)
        file["/Estop/narrow/count"] = np.zeros(int(1e5), dtype=np.uint64)

        file["/param/distribution_v0/seed"] = int(1)
        file["/param/distribution_v0/name"] = "normal"
        file["/param/distribution_v0/param"] = [250.0, 10.0]

        file["/param/distribution_w/seed"] = int(0)

        if args.distribution == "power":
            file["/param/distribution_w/name"] = "power"
            file["/param/distribution_w/param"] = [float(args.k), 1e-6]
        elif args.distribution == "weibull":
            file["/param/distribution_w/name"] = "weibull"
            file["/param/distribution_w/param"] = [float(args.k), 1.0, 1e-6]
        elif args.distribution == "random":
            file["/param/distribution_w/name"] = "random"
            file["/param/distribution_w/param"] = [1.0, 1e-6]

        if args.fext is not None:
            file["/param/distribution_f/seed"] = int(2)
            file["/param/distribution_f/name"] = "normal"
            file["/param/distribution_f/param"] = [0, args.fext]

        file["/param/m"] = float(1)
        file["/param/eta"] = float(0.1)
        file["/param/mu"] = float(1)

        file["/meta/version_dependencies"] = deps
        file["/meta/version_compiler"] = apts.version_compiler()

        file.flush()

    hist_w_log = enstat.histogram(
        bin_edges=file["/wstop/log/bin_edges"][...],
        count=file["/wstop/log/count"][...],
    )

    hist_v_log = enstat.histogram(
        bin_edges=file["/vstop/log/bin_edges"][...],
        count=file["/vstop/log/count"][...],
    )

    hist_E_log = enstat.histogram(
        bin_edges=file["/Estop/log/bin_edges"][...],
        count=file["/Estop/log/count"][...],
    )

    hist_w_broad = enstat.histogram(
        bin_edges=file["/wstop/broad/bin_edges"][...],
        count=file["/wstop/broad/count"][...],
    )

    hist_v_broad = enstat.histogram(
        bin_edges=file["/vstop/broad/bin_edges"][...],
        count=file["/vstop/broad/count"][...],
    )

    hist_E_broad = enstat.histogram(
        bin_edges=file["/Estop/broad/bin_edges"][...],
        count=file["/Estop/broad/count"][...],
    )

    hist_w_narrow = enstat.histogram(
        bin_edges=file["/wstop/narrow/bin_edges"][...],
        count=file["/wstop/narrow/count"][...],
    )

    hist_v_narrow = enstat.histogram(
        bin_edges=file["/vstop/narrow/bin_edges"][...],
        count=file["/vstop/narrow/count"][...],
    )

    hist_E_narrow = enstat.histogram(
        bin_edges=file["/Estop/narrow/bin_edges"][...],
        count=file["/Estop/narrow/count"][...],
    )

    seed_w = file["/param/distribution_w/seed"][...]
    parameters_w = file["/param/distribution_w/param"][...]

    seed_v = file["/param/distribution_v0/seed"][...]
    parameters_v = file["/param/distribution_v0/param"][...]

    if str(file["/param/distribution_w/name"].asstr()[...]) == "random":
        distribution_w = prrng.distribution.random
    elif str(file["/param/distribution_w/name"].asstr()[...]) == "power":
        distribution_w = prrng.distribution.power
    elif str(file["/param/distribution_w/name"].asstr()[...]) == "weibull":
        distribution_w = prrng.distribution.weibull
    else:
        raise ValueError("Unknown distribution")

    if "distribution_f" == file["param"]:
        fext = True
        seed_f = file["/param/distribution_f/seed"][...]
        parameters_f = file["/param/distribution_f/param"][...]
    else:
        fext = False

    m = file["/param/m"][...]
    eta = file["/param/eta"][...]
    mu = file["/param/mu"][...]

    for batch in tqdm.tqdm(range(file["/batch"][...], file["/batch"][...] + nbatch)):

        v0 = prrng.pcg32(batch * ngroup + seed_v, 0).normal([ngroup], *parameters_v)

        if fext:
            wstop, vstop = apts.throw_particle_Quadratic_tilted(
                distribution_w=distribution_w,
                parameters_w=parameters_w,
                distribution_f=prrng.distribution.normal,
                parameters_f=parameters_f,
                v0=v0,
                m=m,
                eta=eta,
                mu=mu,
                seed_w=batch * ngroup + seed_w,
                seed_f=batch * ngroup + seed_f,
            )
        else:
            wstop, vstop = apts.throw_particle_Quadratic(
                distribution_w=distribution_w,
                parameters_w=parameters_w,
                v0=v0,
                m=m,
                eta=eta,
                mu=mu,
                f=0,
                seed=batch * ngroup + seed_w,
            )

        hist_w_log += wstop
        hist_v_log += vstop
        hist_E_log += 0.5 * m * vstop**2

        hist_w_broad += wstop
        hist_v_broad += vstop
        hist_E_broad += 0.5 * m * vstop**2

        hist_w_narrow += wstop
        hist_v_narrow += vstop
        hist_E_narrow += 0.5 * m * vstop**2

        file["/wstop/log/count"][...] = hist_w_log.count
        file["/vstop/log/count"][...] = hist_v_log.count
        file["/Estop/log/count"][...] = hist_E_log.count

        file["/wstop/broad/count"][...] = hist_w_broad.count
        file["/vstop/broad/count"][...] = hist_v_broad.count
        file["/Estop/broad/count"][...] = hist_E_broad.count

        file["/wstop/narrow/count"][...] = hist_w_narrow.count
        file["/vstop/narrow/count"][...] = hist_v_narrow.count
        file["/Estop/narrow/count"][...] = hist_E_narrow.count

        file["/batch"][...] = batch + 1

        file.flush()
