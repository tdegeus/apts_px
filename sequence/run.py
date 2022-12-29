import pathlib

import apts
import h5py
import prrng

root = pathlib.Path(__file__).parent
basename = pathlib.Path(__file__).stem

with h5py.File(root / "data.h5", "w") as file:

    file["/param/m"] = 1
    file["/param/eta"] = 0.1
    file["/param/mu"] = 1
    file["/param/f"] = 0

    distribution = prrng.distribution.weibull
    parameters = [2]

    for seed, v0 in enumerate(prrng.pcg32(0).normal([100], 100, 10)):

        details = apts.ThrowParticleQuadratic(
            distribution_w=distribution,
            parameters_w=parameters,
            v0=v0,
            seed=seed,
            m=file["/param/m"][...],
            eta=file["/param/eta"][...],
            mu=file["/param/mu"][...],
            f=file["/param/f"][...],
        )

        file[f"/seed/{seed:d}/w"] = details.w
        file[f"/seed/{seed:d}/v0"] = details.v0
        file[f"/seed/{seed:d}/tau_exit"] = details.tau_exit
