# Get 1D mesh
from netgen.meshing import *
import ngsolve
import numpy as np

def generate_1d_mesh_dirichlet(N_element: int):
    # generate a 1D mesh
    m = Mesh(dim=1)

    # add points
    pnums = []
    for i in range(0, N_element + 1):
        pnums.append(m.Add(MeshPoint(Pnt(2 * np.pi * i / N_element, 0, 0))))

    # add element
    idx = m.AddRegion("medium", dim=1)
    for i in range(0, N_element):
        m.Add(Element1D([pnums[i], pnums[i + 1]], index=idx))

    # add boundary condition
    idx_left = m.AddRegion("left", dim=0)
    idx_right = m.AddRegion("right", dim=0)

    m.Add(Element0D(pnums[0], index=idx_left))
    m.Add(Element0D(pnums[N_element], index=idx_right))

    # create mesh
    mesh = ngsolve.Mesh(m)
    return mesh


def generate_1d_mesh_periodic(N_element: int):
    # generate a 1D mesh
    m = Mesh(dim=1)

    # add points
    pnums = []
    for i in range(0, N_element + 1):
        pnums.append(m.Add(MeshPoint(Pnt(2 * np.pi * i / N_element, 0, 0))))

    # add element
    idx = m.AddRegion("medium", dim=1)
    for i in range(0, N_element):
        m.Add(Element1D([pnums[i], pnums[i + 1]], index=idx))

    m.Add(Element0D(pnums[0], index=1))
    m.Add(Element0D(pnums[N_element], index=2))
    m.AddPointIdentification(pnums[0], pnums[N_element], 1, 2)

    # create mesh
    mesh = ngsolve.Mesh(m)
    return mesh
