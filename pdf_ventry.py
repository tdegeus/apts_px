import matplotlib.pyplot as plt
import numpy as np
import apts
import GooseMPL as gplt

plt.style.use(["goose", "goose-latex", "goose-autolayout"])


# class Potential:
#     """
#     Solution valid within a potential.
#     Assume that gamma = 0 for simplicity.
#     """

#     def __init__(self, v0=0, w=1, f=0, m=1, eta=0.1, mu=1):
#         """
#         :param v0: Initial velocity.
#         :param w: Width of the potential.
#         :param f: Tilt: external force.
#         :param m: Mass.
#         :param eta: Damping coefficient.
#         :param mu: Stiffness of the potential.
#         """

#         self.w = w
#         self.eta = eta
#         self.m = m
#         self.kappa = mu
#         self.F = f

#         self.delta_r = self.F / self.kappa
#         self.r0 = -self.w / 2
#         self.rmax = self.w / 2
#         self.r0prime = self.r0 - self.delta_r

#         self.labda = self.eta / (2 * self.m)
#         self.omega = np.sqrt(self.kappa / self.m - (self.eta / (2 * self.m)) ** 2)

#         self.v0 = v0
#         self.phase = np.arctan(self.omega / self.labda)

#     @property
#     def v0(self):
#         """
#         Get the currently set initial velocity.
#         """

#         return self._v0

#     @v0.setter
#     def v0(self, value):
#         """
#         Update the initial velocity.
#         """

#         self._v0 = value
#         self.v0prime = self._v0
#         self.alpha = 0.5 * (
#             self.r0prime - 1j * (self.labda * self.r0prime + self.v0prime) / self.omega
#         )

#         if np.real(self.alpha) >= 0:
#             self.chi = 0
#         else:
#             if np.imag(self.alpha) >= 0:
#                 self.chi = 1
#             else:
#                 self.chi = -1

#         print(self.chi)

#         self.phi = self.chi * np.pi - np.arctan(
#             self.labda / self.omega + self.v0prime / (self.omega * self.r0prime)
#         )
#         self.L = np.sqrt(
#             ((self.omega * self.r0prime) ** 2 + (self.labda * self.r0prime + self.v0prime) ** 2)
#             / self.omega**2
#         )
#         self.Lprime = self.L * self.labda * np.sqrt(1 + (self.omega / self.labda) ** 2)

#     def r(self, tau):
#         """
#         Get the particle's position as different times.
#         """

#         return (
#             self.L * np.exp(-self.labda * tau) * np.cos(self.omega * tau + self.phi) + self.delta_r
#         )

#     def v(self, tau):
#         """
#         Get the particle's velocity at different times.
#         """

#         return (
#             -self.Lprime
#             * np.exp(-self.labda * tau)
#             * np.cos(self.omega * tau + self.phi - self.phase)
#         )

#     @property
#     def tau_at_rmax(self):
#         """
#         Get the time taun such that v(taun) = 0 for the first time.
#         This corresponds for the maximum position that the particle can have
#         (ignoring that this position may be outside of the potential).
#         """

#         n = np.array([-1, 1])
#         taun = ((n + 0.5) * np.pi + self.phase - self.phi) / self.omega
#         return np.min(taun[taun >= 0])

#     @property
#     def exits(self):
#         """
#         Return ``True`` is the particle will exit the well.
#         """

#         return self.r(self.tau_at_rmax) > self.rmax

#     @property
#     def tau_exit(self):
#         """
#         Time at exit.
#         """

#         assert self.exits

#         t0 = 0
#         t1 = self.tau_at_rmax

#         for i in range(20):

#             tau = np.linspace(t0, t1, 100)
#             pos = self.r(tau)
#             i = np.argmax(pos >= self.rmax)
#             t0 = tau[i - 1]
#             t1 = tau[i]

#             if (t1 - t0) / t1 < 1e-5:
#                 return 0.5 * (t0 + t1)

#         raise OSError("No convergence found")


# fig, ax = plt.subplots()

# particle = apts.Quadratic(v0 = 1)
# tau = np.linspace(0, 10, 1000)

# print(particle.r(tau))

# ax.plot(tau, particle.r(tau))
# ax.plot(tau, particle.v(tau))

# plt.show()



# eta = np.logspace(-3, 0, 100)
# m = np.logspace(-3, 2, 100)

# d = np.NaN * np.ones((eta.size, m.size))

# for ieta in range(eta.size):
#     for im in range(m.size):

#         if eta[ieta] ** 2 > 4 * m[im]:
#             continue

#         d[ieta, im] = apts.Quadratic(w=1, eta=eta[ieta], m=m[im]).vc

# fig, ax = plt.subplots()
# plt.imshow(d, cmap="jet")
# plt.show()


# # # w = np.logspace(-6, 0, 100)
# # # w = np.linspace(0, 1, 200)[1:]
w = np.linspace(0, 10, 1000)[1:]
# v = np.logspace(-4, -0.8, 1000)
v = np.linspace(1e-3, 1, 1000)
d = np.empty((w.size, v.size))

for i in range(w.size):

    particle = apts.Quadratic(w=w[i])

    for j in range(v.size):

        particle.v0 = v[j]

        if not particle.exits:
            d[i, j] = np.NaN
        else:
            d[i, j] = v[j] - particle.v(particle.tau_exit)

fig, axes = gplt.subplots(ncols=2)
# # # ax.imshow(d, extent=[v[0], v[-1], w[0], w[-1]])
ax = axes[0]
ax.contourf(v, w, d, levels=10)

# for i in 0.01 * np.arange(8):

#     x = np.array([i, i + 0.02])
#     y = 10 * x
#     y -= y[0]

#     ax.plot(x, y, "k")

ax.set_xlabel(r"$v$")
ax.set_ylabel(r"$w$")

ax = axes[1]
d = d.ravel()
d = d[~np.isnan(d)]
bin_edges = gplt.histogram_bin_edges(d, bins=1000)
P, x = gplt.histogram(d, bins=bin_edges, return_edges=False, density=True)

ax.plot(x, P, marker=".")

# ax.set_xlabel(r"$v_\mathrm{exit}$")
# ax.set_ylabel(r"$P(v_\mathrm{exit})$")

bin_edges = gplt.histogram_bin_edges(v, bins=100)
P, x = gplt.histogram(d, bins=bin_edges, return_edges=False, density=True)

ax.plot(x, P, marker=".", c="r")


# ax.set_xlim([0, 2])

ax.set_xscale("log")
ax.set_yscale("log")

# gplt.plot_powerlaw(1, 0.1, 0, 1, axis=ax, c="b")

plt.show()

# # # d = [Potential(i for i in w]


# # # fig, ax = plt.subplots()

# # # tau = np.linspace(0, 10, 1000)
# # # pot = Potential(1)

# # # print(pot.exits)
# # # pot.tau_exit

# # # ax.plot(tau, pot.r(tau), label=r"$r$")
# # # ax.plot(tau, pot.v(tau), label=r"$v$")
# # # ax.axvline(pot.tau_at_rmax)
# # # ax.legend()
# # # plt.show()
