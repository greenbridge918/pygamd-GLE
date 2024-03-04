from pygamd import chare
import numpy as np
import numba as nb
from numba import cuda
from pygamd.forces import potentials
import math

@cuda.jit(device=True)
def wrap(n, box):
    return n - int(box * math.floor(n/box))

@cuda.jit("void(int32[:], float32[:, :], float32[:, :, :], float32[:, :, :], float32[:], float32[:], int32, int32, float32, int32, float32)")
def cu_convolution(member, force, memory, white_noise, kernel, rfcoef, n_update, n_itg, temperature, seed, dt):
    i = cuda.grid(1)
    if i > member.shape[0]:
        return
    idx = member[i]
    fi = force[idx]
    current_ts = wrap(n_update - 1, n_itg)
    noise_centerl = n_itg - 1
    for j in range(n_itg):
        t = wrap(current_ts - j, n_itg)
        k = kernel[j]
        v = memory[t, idx]
        fi[0] -= k * v[0] * dt
        fi[1] -= k * v[1] * dt
        fi[2] -= k * v[2] * dt

        frt = wrap(current_ts + noise_centerl + j, n_itg*2-1)
        brt = wrap(current_ts + noise_centerl - j, n_itg*2-1)
        faj = rfcoef[noise_centerl+j]
        baj = rfcoef[noise_centerl-j]
        Wf = white_noise[frt, idx]
        Wb = white_noise[brt, idx]
        fi[0] += math.sqrt(temperature) * (faj * Wf[0] + baj * Wb[0])
        fi[1] += math.sqrt(temperature) * (faj * Wf[1] + baj * Wb[1])
        fi[2] += math.sqrt(temperature) * (faj * Wf[2] + baj * Wb[2])
    fi[0] -= math.sqrt(temperature) * rfcoef[noise_centerl] * white_noise[wrap(current_ts + noise_centerl, n_itg*2-1), idx][0]
    fi[1] -= math.sqrt(temperature) * rfcoef[noise_centerl] * white_noise[wrap(current_ts + noise_centerl, n_itg*2-1), idx][1]
    fi[2] -= math.sqrt(temperature) * rfcoef[noise_centerl] * white_noise[wrap(current_ts + noise_centerl, n_itg*2-1), idx][2]

    white_noise[wrap(current_ts + noise_centerl, n_itg*2-1), idx] = potentials.rng.saruprng(nb.int32(seed), nb.int32(idx), nb.int32(seed+1), nb.int32(1), nb.float32(-1.0), nb.float32(1.0))
    white_noise[wrap(current_ts + noise_centerl, n_itg*2-1), idx] = potentials.rng.saruprng(nb.int32(seed), nb.int32(seed+2), nb.int32(idx), nb.int32(1), nb.float32(-1.0), nb.float32(1.0))
    white_noise[wrap(current_ts + noise_centerl, n_itg*2-1), idx] = potentials.rng.saruprng(nb.int32(seed+3), nb.int32(idx), nb.int32(seed), nb.int32(1), nb.float32(-1.0), nb.float32(1.0))


class GLE:
    def __init__(self, info, group, Kernel, RfCoef, temperature, seed, n_itg):
        assert Kernel.shape[0] == n_itg, "The length of Kernel must match the integration steps."
        self.info = info
        self.ps = info.find_particle_set(group)
        if self.ps is None:
            self.ps = chare.particle_set(info, group)
            info.particle_set.append(self.ps)
        self.Kernel = Kernel
        self.RfCoef = RfCoef
        self.dKernel = cuda.to_device(self.Kernel)
        self.dRfCoef = cuda.to_device(self.RfCoef)
        self.temperature = temperature
        self.seed = seed
        self.n_itg = n_itg
        self.memory_vel = np.zeros((self.n_itg, self.ps.nme, 3), dtype=np.float32)
        self.white_noise = np.random.randn(self.n_itg*2-1, self.ps.nme, 3).astype(np.float32)
        #self.white_noise = np.zeros((self.n_itg, self.ps.nme, 3), dtype=np.float32)
        self.d_memory_vel = cuda.to_device(self.memory_vel)
        self.d_white_noise = cuda.to_device(self.white_noise)
        self.n_update = np.int32(0)
        self.block_size=64
        self.nblocks = math.ceil(self.ps.nme / self.block_size)
        self.name = "force"

    def compute(self, timestep):
        ## first update the memory velocity array
        self.d_memory_vel[self.n_update] = self.info.d_vel[:,:3]
        #print(self.info.d_force.copy_to_host())
        self.n_update += 1
        if self.n_update == self.n_itg:
            self.n_update = 0
        ## begin time convolution
        cu_convolution[self.nblocks, self.block_size](self.ps.d_member, self.info.d_force, self.d_memory_vel, 
                        self.d_white_noise, self.dKernel, self.dRfCoef, self.n_update, self.n_itg, self.temperature, self.seed, self.info.dt)
        


