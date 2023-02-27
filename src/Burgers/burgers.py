import matplotlib.pyplot as plt
import numpy as np
from ngsolve import *
from ngsolve.meshes import Make1DMesh
from tqdm import tqdm

# mesh = generate_1d_mesh_periodic(25)
xL, xR = 0, 2 * np.pi
N_mesh = 100
mesh = Make1DMesh(N_mesh, mapping=lambda x: xL + (xR - xL) * x, periodic=True)

# normal direction
n = specialcf.normal(mesh.dim)

# PDE
f = lambda u: u**2 / 2
epsilon = 0.01

# DG space
order = 3
V = L2(mesh, order=order)

# define trial- and test-functions
u = V.TrialFunction()
v = V.TrialFunction()

xi = V.TestFunction()
eta = V.TestFunction()

# initial condition
u0 = GridFunction(V)
u0.Set(sin(x))

# time stepping parameters
tau = 0.1 * (2 * np.pi / N_mesh) ** 2
alpha = 1


def euler_forward_step(gfu: GridFunction):
    # solve v
    B_v = BilinearForm(V)
    B_v += SymbolicBFI(-u * grad(eta))
    flux_u = 1 / 2 * (u + u.Other())
    B_v += SymbolicBFI(flux_u * eta * n, element_boundary=True)
    gfv = GridFunction(V, "v")
    result_v = gfv.vec.CreateVector()
    B_v.Apply(gfu.vec, result_v)
    V.SolveM(result_v, 1.0)
    gfv.vec.data = result_v

    # solve u
    gfu_next = GridFunction(V)
    result_u = gfu_next.vec.CreateVector()

    A_u = BilinearForm(V)
    A_u += SymbolicBFI(f(u) * grad(xi))
    flux_f = 1 / 2 * (f(u) + f(u.Other()) - alpha * (u.Other() - u) * n)
    A_u += SymbolicBFI(-flux_f * xi * n, element_boundary=True)
    A_u.Apply(gfu.vec, result_u)
    V.SolveM(result_u, 1.0)
    gfu_next.vec.data = gfu.vec.data + tau * result_u

    A_v = BilinearForm(V)
    A_v += SymbolicBFI(-v * grad(xi))
    flux_v = 1 / 2 * (v + v.Other())
    A_v += SymbolicBFI(flux_v * xi * n, element_boundary=True)
    result_u_v = gfu_next.vec.CreateVector()
    A_v.Apply(gfv.vec, result_u_v)
    V.SolveM(result_u_v, 1.0)
    gfu_next.vec.data += epsilon * tau * result_u_v

    return gfu_next


def ssp_rk3(u_fn):
    f1 = euler_forward_step(u_fn)
    f2 = 3 / 4 * u_fn + 1 / 4 * euler_forward_step(f1)
    u2 = GridFunction(V)
    u2.Set(f2)
    f3 = 1 / 3 * u_fn + 2 / 3 * euler_forward_step(u2)
    u3 = GridFunction(V)
    u3.Set(f3)
    return u3


evaluation_points = np.linspace(0, 2 * np.pi, 101).reshape(-1, 1)


def evaluate(gfu: GridFunction):
    vals = [gfu(mesh(*p)) for p in evaluation_points]
    return vals


u_fn = u0
for i in tqdm(range(int(1.5 / tau))):
    u_fn = ssp_rk3(u_fn)
    if i % 10 == 0:
        u_vals = evaluate(u_fn)
        if np.isnan(u_vals).any():
            print("NaN at epoch", i)
            break


u_vals = evaluate(u_fn)
plt.plot(evaluation_points, u_vals)
plt.ylim(-1.25, 1.25)
plt.tight_layout()
plt.savefig("./test/burgers_result.png", dpi=300)
np.savetxt(f"./test/u_viscid.csv", u_vals, delimiter=",")  # _{order}_{N_mesh}
