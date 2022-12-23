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

with h5py.File(args.file, "a" if args.append else "w") as file:

    if not args.append:

        file["/batch"] = int(0)

        file[f"/wstop/bin_edges"] = np.linspace(1e-6, 5e1, int(1e5) + 1)
        file[f"/vstop/bin_edges"] = np.linspace(1e-6, 5e1, int(1e5) + 1)
        file[f"/wstop/count"] = np.zeros(int(1e5), dtype=np.uint64)
        file[f"/vstop/count"] = np.zeros(int(1e5), dtype=np.uint64)

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

        file["/meta/version_dependencies"] = apts.version_dependencies()
        file["/meta/version_compiler"] = apts.version_compiler()

        file.flush()

    hist_w = enstat.histogram(
        bin_edges=file["/wstop/bin_edges"][...],
        count=file["/wstop/count"][...],
    )

    hist_v = enstat.histogram(
        bin_edges=file["/vstop/bin_edges"][...],
        count=file["/vstop/count"][...],
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

        hist_w += wstop
        file["/wstop/count"][...] = hist_w.count

        hist_v += vstop
        file["/vstop/count"][...] = hist_v.count

        file["/batch"][...] = batch + 1

        file.flush()
