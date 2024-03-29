'''
PYGAMD - Python GPU-Accelerated Molecular Dynamics Software
VERSION 1
COPYRIGHT
	PYGAMD Copyright (c) (2021) You-Liang Zhu, Zhong-Yuan Lu
LICENSE
	This program is a free software: you can redistribute it and/or 
	modify it under the terms of the GNU General Public License. 
	This program is distributed in the hope that it will be useful, 
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANT ABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
	See the General Public License v3 for more details.
	You should have received a copy of the GNU General Public License
	along with this program. If not, see <http://www.gnu.org/licenses/>.
DISCLAIMER
	The authors of PYGAMD do not guarantee that this program and its 
	derivatives are free from error. In no event shall the copyright 
	holder or contributors be liable for any indirect, incidental, 
	special, exemplary, or consequential loss or damage that results 
	from its use. We also have no responsibility for providing the 
	service of functional extension of this program to general users.
USER OBLIGATION 
	If any results obtained with PYGAMD are published in the scientific 
	literature, the users have an obligation to distribute this program 
	and acknowledge our efforts by citing the paper "Y.-L. Zhu, H. Liu, 
	Z.-W. Li, H.-J. Qian, G. Milano, and Z.-Y. Lu, J. Comput. Chem. 2013,
	34, 2197-2211" in their article.
CORRESPONDENCE
	Dr. You-Liang Zhu
	Email: ylzhu@pygamd.com
'''

from pygamd import chare
import numpy as np
import numba as nb
from numba import cuda
from pygamd.forces import potentials
import pygamd.snapshots.box as box_func
import math

@cuda.jit(device=True)
def wrap(n, box):
    return n - int(box * math.floor(n/box))
    """
    one-dimensional pbc
    """

@cuda.jit("void(int32, int32[:], float32[:, :], float32[:, :], float32[:, :], int32[:, :], float32[:], float32, int32, float32[:,:,:], int32)")
def cu_first_step(nme, member, pos, vel, force, image, box, dt, n_update, memory, n_itg):
    i = cuda.grid(1)
    if i < nme:
        idx = member[i]
        pi = pos[idx]
        vi = vel[idx]
        mi = vi[3]
        ai = force[idx]
        ii = image[idx]
    
        ai[0] /= mi
        ai[1] /= mi
        ai[2] /= mi

        # update half-step velocity
        vi[0] = (vi[0] + nb.float32(0.5)*ai[0]*dt)
        vi[1] = (vi[1] + nb.float32(0.5)*ai[1]*dt)
        vi[2] = (vi[2] + nb.float32(0.5)*ai[2]*dt)
    
        # update a step of position
        pi[0] += vi[0]*dt
        pi[1] += vi[1]*dt				
        pi[2] += vi[2]*dt
    
        # apply pbc
        box_func.cu_box_wrap(pi, box, ii)
        pos[idx][0] = pi[0]
        pos[idx][1] = pi[1]
        pos[idx][2] = pi[2]
    
        vel[idx][0] = vi[0]
        vel[idx][1] = vi[1]
        vel[idx][2] = vi[2]
    
        image[idx][0] = ii[0]
        image[idx][1] = ii[1]
        image[idx][2] = ii[2]

        # store velocity of this timestep
        v_up = n_update
        memory[v_up, idx, 0] = vi[0]
        memory[v_up, idx, 1] = vi[1]
        memory[v_up, idx, 2] = vi[2]
        



@cuda.jit("void(int32[:], float32[:, :], float32[:, :], float32[:, :, :], float32[:, :, :], float32[:], float32[:], int32, int32, float32, int32, float32, int32)")
def cu_convolution(member, vel, force, memory, white_noise, kernel, rfcoef, n_update, n_itg, temperature, seed, dt, n_update_w):
    i = cuda.grid(1)
    if i >= member.shape[0]:
        return
    idx = member[i]
    vi = vel[idx]
    mi = vi[3]
    fi = force[idx]

    current_ts = n_update
    current_ts_w = n_update_w
    accumx = 0
    accumy = 0
    accumz = 0

    # The dissipative and random forces share n_itg*2-1 loops but only n_itg loops are used in calculation of dissipative forces.
    for j in range(n_itg*2-1):
        if j < n_itg:
            t = wrap(current_ts - j, n_itg)
            k = kernel[j]
            v = memory[t, idx]
            # m is a scale factor and only is 0.5 when t=0. See 10.1021/acs.jctc.7b00274.
            m = 1
            if j == 0:
                m = 0.5
            # Add dissipative force into total force
            fi[0] -= m * k * v[0] * dt
            fi[1] -= m * k * v[1] * dt
            fi[2] -= m * k * v[2] * dt

    # Calculate the random force using colored noise coefficient multiply white noise
        rt = wrap(current_ts_w + j, n_itg*2-1)
        aj = rfcoef[j]
        W = white_noise[rt, idx]

        accumx += aj * W[0]
        accumy += aj * W[1]
        accumz += aj * W[2]

    # Add random force into total force
    fi[0] += math.sqrt(3*temperature) * accumx
    fi[1] += math.sqrt(3*temperature) * accumy
    fi[2] += math.sqrt(3*temperature) * accumz

    up = wrap(current_ts_w + 1 , 2*n_itg-1)

    # Update the white noise
    white_noise[up, idx, 0] = potentials.rng.saruprng(nb.int32(seed), nb.int32(idx), nb.int32(seed+1), nb.int32(1), nb.float32(-1.0), nb.float32(1.0))
    white_noise[up, idx, 1] = potentials.rng.saruprng(nb.int32(seed), nb.int32(seed+2), nb.int32(idx), nb.int32(1), nb.float32(-1.0), nb.float32(1.0))
    white_noise[up, idx, 2] = potentials.rng.saruprng(nb.int32(seed+3), nb.int32(idx), nb.int32(seed), nb.int32(1), nb.float32(-1.0), nb.float32(1.0))

    m_inv = nb.float32(1.0) / mi

    # Update half-step velocity according to new force
    vi[0] = (vi[0] + nb.float32(0.5)*fi[0]*m_inv*dt)
    vi[1] = (vi[1] + nb.float32(0.5)*fi[1]*m_inv*dt)
    vi[2] = (vi[2] + nb.float32(0.5)*fi[2]*m_inv*dt)
		
    vel[idx][0] = vi[0]
    vel[idx][1] = vi[1]
    vel[idx][2] = vi[2]

    force[idx][0] = fi[0]
    force[idx][1] = fi[1]
    force[idx][2] = fi[2]

    v_up = wrap(n_update, n_itg)
    memory[v_up, idx, 0] = vi[0]
    memory[v_up, idx, 1] = vi[1]
    memory[v_up, idx, 2] = vi[2]


class gle:
    """ 
    The GLE integrator splits a timestep into two steps: 1. Update the half step velocity and then update the whole position;
    2. update the force and the remain half step velocity.
    """
    def __init__(self, info, group, ci, Kernel, RfCoef, temperature, seed, n_itg):
        """
        info: All particle infomation, such as velocity, force and position
        group: A group of particles in which the same integrator is used
        Kernel: Memory kernel
        RfCoef: Colored noise coefficient
        temperature: k_B T
        seed: random seed
        n_itg: the number of numerical points of memory kernel
        memory_vel: the stored velocities of the last $n_itg$ steps
        white_noise: the stored series of white noises with the size (2*n_itg - 1)
        n_update: index in the memory_vel array corresponding to the current timestep
        n_update_w: index in the white_noise corresponding to the current timestep
        """
        assert Kernel.shape[0] == n_itg, "The length of Kernel must match the number of numerical points"
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
        self.white_noise = np.zeros((self.n_itg*2-1, self.ps.nme, 3), dtype=np.float32)
        self.d_memory_vel = cuda.to_device(self.memory_vel)
        self.d_white_noise = cuda.to_device(self.white_noise)
        self.n_update = np.int32(0)
        self.n_update_w = np.int32(0)
        self.block_size=64
        self.nblocks = math.ceil(self.ps.nme / self.block_size)
        self.name = "integration"

    def firststep(self, timestep):
        cu_first_step[self.nblocks, self.block_size](self.ps.nme, self.ps.d_member, self.info.d_pos, self.info.d_vel, self.info.d_force, 
                                                self.info.d_image, self.info.d_box, self.info.dt, self.n_update, self.d_memory_vel, self.n_itg)
    def secondstep(self, timestep):
        cu_convolution[self.nblocks, self.block_size](self.ps.d_member, self.info.d_vel, self.info.d_force, self.d_memory_vel, 
                            self.d_white_noise, self.dKernel, self.dRfCoef, self.n_update, self.n_itg, self.temperature, self.seed+timestep, self.info.dt, self.n_update_w, timestep) 
        # An one-dimensional pbc array is used to store velocities and white noise, i.e., i_{n_itg} == i_{0}
        # The length of stored white noise is 2*n_itg - 1 because of the symmetry of the Fourier transform
        self.n_update += 1
        if self.n_update == self.n_itg:
            self.n_update = 0
        self.n_update_w += 1
        if self.n_update_w == self.n_itg*2-1:
            self.n_update_w = 0
    

    def register(self, timestep):
        """
        update system infomation
        """
        self.info.compute_properties['temperature']=True		
        self.info.compute_properties['pressure']=True		
        self.info.compute_properties['potential']=True		
        self.info.compute_properties['momentum']=True



