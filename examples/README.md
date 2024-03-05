# Examples
## Numerical Potential
The implementation of numerical (table) potential is based on self-defined device functions that could be written in script and conveyed to kernel funciton for calculation.
In order to use the numerical potential, one should import table_reader for initilizing cubic spline parameters,
```
from pygamd.forces.table_reader import table_reader
```
Build force class, select function type as pair_table/bond_table/angle_table, and set parameters,
```
fn = pygamd.force.nonbonded(info=mst, rcut=1.75, func="pair_table", exclusion=["bond", "angle"])
fn.setParams(type_i="P", type_j="P", param=table_reader.get_parameters("pair.table", n_points=1751, rcut=1.750))
app.add(fn)

fb = pygamd.force.bond(info=mst, func="bond_table")
fb.setParams(bond_type = 'P-P', param=table_reader.get_parameters("bond.table", n_points=2000, rcut=1.999))
app.add(fb)

fa = pygamd.force.angle(info=mst, func="angle_table")
fa.setParams(angle_type = 'P-P-P', param=table_reader.get_parameters("angle.table", n_points=2000, rcut=np.pi))
app.add(fa)
```
The potentials should have two columns in a file, such as for pair potential
```
r Potential
... ...
```
The distance or angle points in first column should be in equal interval and the potentials at the corresponding points are given in second column. For angle potential, use rad rather than degree.
## GLE Integrator
The GLE integrator can be used to integrate with memory kernel and control temperature. 
```
gle = pygamd.integration.gle(info=mst, group="all", Kernel=Kernel, RfCoef=RfCoef, temperature=Temperature, seed=210, n_itg=n_integration)
app.add(gle)
```
where Kernel presents memory kernel, RfCoef is colored noise coefficients and n_itg is number of numerical points of memory kernel. Fourier transform is typically used to get colored noise coefficients.
```
def ft(f, t):
    w = 2 * np.pi * np.fft.fftfreq(len(f)) / (t[1] - t[0])
    g = np.fft.fft(f)
    g *= (t[1] - t[0]) * np.exp(-complex(0, 1) * w * t[0])
    return g, w


def ift(f, w, t):
    f *= np.exp(complex(0, 1) * w * t[0])
    g = np.fft.ifft(f)
    g *= len(f) * (w[1] - w[0]) / (2 * np.pi)
    return g

def ColoredNoiseCoefficient(kernel, t):
    kernel_sym = np.concatenate((kernel[:0:-1], kernel))
    t_sym = np.concatenate((-t[:0:-1], t))
    kernel_ft, w = ft(kernel_sym, t_sym)
    sqk_ft = np.sqrt(kernel_ft)
    sqk = ift(sqk_ft, w, t_sym).real
    return sqk

t = np.arange(n_integration) * dt
RfCoef = ColoredNoiseCoefficient(Kernel, t).astype(np.float32) * np.sqrt(dt)
```
## Iterative Memory Reconstruction Velocity-Autocorrelation (IMRV)
Develop memory kernel from velocity-autocorrelation of atomistic simulations. See articles to find background and details.

First you should ensure the `template` located at the same directory with `IMRV.py`, and required these files in `template`: *.table (conservative interactions with numerical form), 
coor.mst (initial configuration), md.py (md script), cal_vacf.py (calculate VACF) and AA_VACF.txt (reference VACF).

Then define IMRV class,
```
imrv = IMRV("template/AA_VACF.txt", n_itgs=400, n_iteration=0, max_iteration=200, dt=0.005, mass=104, temperature=4.16, gamma=1780, order=0, gpuid=0)
imrv.BeginIteration()
```
n_itgs: the number of numerical points of memory kernel; n_iteration: initial number of iteration; max_iteration: maximum number of further iterations; dt: timestep; mass: mass of CG bead; temperature: k_BT; gamma: integrated value of memory kernel; order: the orders of derivatives of the VACF in the iterative term, $VACF_{AA} - VACF_{GLE}$ is 0 and  $\frac{\partial(VACF_{AA} - VACF_{GLE})}{\partial t}$ is 1; gpuid: which GPU perform MD simulations.
















