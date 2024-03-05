import pygamd
from numba import cuda
import numba as nb
import numpy as np
from pygamd.forces.table_reader import table_reader

mst = pygamd.snapshot.read("coor.mst")
timestep = 0.005 #time unit 1tau = 1ps
timestep = timestep / 10 ## using mass unit 1m=104amu
app = pygamd.application.dynamics(info=mst, dt=timestep)

fn = pygamd.force.nonbonded(info=mst, rcut=1.75, func="pair_table", exclusion=["bond", "angle"])
fn.setParams(type_i="P", type_j="P", param=table_reader.get_parameters("pair.table", n_points=1751, rcut=1.750))
app.add(fn)

fb = pygamd.force.bond(info=mst, func="bond_table")
fb.setParams(bond_type = 'P-P', param=table_reader.get_parameters("bond.table", n_points=2000, rcut=1.999))
app.add(fb)

fa = pygamd.force.angle(info=mst, func="angle_table")
fa.setParams(angle_type = 'P-P-P', param=table_reader.get_parameters("angle.table", n_points=2000, rcut=np.pi))
app.add(fa)

T = 500
Temperature = T / 120.27236
inn = pygamd.integration.npt(info=mst, group='all', method="berendsen", tau=0.5, temperature=Temperature, taup=5.0, pressure=0.061)
app.add(inn)

di = pygamd.dump.data(info=mst, group='all', file='data.log', period=100)
app.add(di)

dd = pygamd.dump.mst(info=mst, group="all", file="output.mst", period=int(1e4), split=True)
app.add(dd)

app.run(int(1e5))
