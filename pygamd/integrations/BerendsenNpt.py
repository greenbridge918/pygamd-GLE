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

from pygamd.forces import potentials
import pygamd.snapshots.box as box_func
import numpy as np
import numba as nb
from numba import cuda
import math 

@cuda.jit("void(int32, int32[:], float32[:, :], float32[:, :], float32[:, :], float32, float32)")
def cu_first_step(nme, member, pos, vel, force, xi, dt):
	i = cuda.grid(1)
	if i < nme:
		idx = member[i]
		pi = pos[idx]
		vi = vel[idx]
		mi = vi[3]
		ai = force[idx]
		
		ai[0] /= mi
		ai[1] /= mi
		ai[2] /= mi
		
		vi[0] = (vi[0] + nb.float32(0.5)*ai[0]*dt)*xi
		vi[1] = (vi[1] + nb.float32(0.5)*ai[1]*dt)*xi
		vi[2] = (vi[2] + nb.float32(0.5)*ai[2]*dt)*xi
		
		pi[0] += vi[0]*dt
		pi[1] += vi[1]*dt				
		pi[2] += vi[2]*dt

		## coor scale
		#pi[0] *= pi[0]*box_len_scale
		#pi[1] *= pi[1]*box_len_scale				
		#pi[2] *= pi[2]*box_len_scale
		
		#box_func.cu_box_wrap(pi, box, ii)
		pos[idx][0] = pi[0]
		pos[idx][1] = pi[1]
		pos[idx][2] = pi[2]
		
		vel[idx][0] = vi[0]
		vel[idx][1] = vi[1]
		vel[idx][2] = vi[2]

def iso_box_scale(info, scale_factor):
	new_box = np.zeros(3, dtype=np.float32)
	for i in range(3):
		new_box[i] = info.d_box[i] * scale_factor
	info.d_box = cuda.to_device(new_box)

@cuda.jit("void(int32, int32[:], float32[:,:], int32[:,:], float32[:], float32)")
def cu_iso_particle_scale(nme, member, pos, image, box, box_len_scale):
	i = cuda.grid(1)
	if i < nme:
		idx = member[i]
		pi = pos[idx]
		ii = image[idx]
		## coor scale
		pi[0] *= box_len_scale
		pi[1] *= box_len_scale				
		pi[2] *= box_len_scale
		
		box_func.cu_box_wrap(pi, box, ii)
		pos[idx][0] = pi[0]
		pos[idx][1] = pi[1]
		pos[idx][2] = pi[2]
		
		image[idx][0] = ii[0]
		image[idx][1] = ii[1]
		image[idx][2] = ii[2]


@cuda.jit("void(int32, int32[:], float32[:, :], float32[:, :], float32, float32)")
def cu_second_step(nme, member, vel, force, xi, dt):
	i = cuda.grid(1)
	if i < member.shape[0]:
		idx = member[i]
		vi = vel[idx]
		ai = force[idx]
		mi = vi[3]
		
		ai[0] /= mi
		ai[1] /= mi
		ai[2] /= mi
		
		vi[0] = (vi[0] + nb.float32(0.5)*ai[0]*dt)*xi
		vi[1] = (vi[1] + nb.float32(0.5)*ai[1]*dt)*xi
		vi[2] = (vi[2] + nb.float32(0.5)*ai[2]*dt)*xi

		vel[idx][0] = vi[0]
		vel[idx][1] = vi[1]
		vel[idx][2] = vi[2]


class BerendsenNpt:
	#define init function
	def __init__(self, info, ci, tau, temp, taup, pres):
		self.info=info
		self.ci=ci
		self.block_size=64
		self.xi=1.0
		self.pi=1.0
		self.tau=tau
		self.temp=temp
		self.taup=taup
		self.pres=pres
	#calculate non-bonded force
	def firststep(self, timestep):
		self.ci.calculate(timestep)
		curr_T = self.ci.temp
		curr_P = self.ci.pressure
		if(curr_T<1.0e-6):
			curr_T=0.0001
		self.xi = (1.0+self.info.dt*(self.temp/curr_T-1.0)/self.tau)**(1/2) 
		self.pi = (1 + self.info.dt*(curr_P - self.pres)/self.taup)**(1/3)
		nblocks = math.ceil(self.ci.ps.nme / self.block_size)
		iso_box_scale(self.info, self.pi)
		cu_first_step[nblocks, self.block_size](self.ci.ps.nme, self.ci.ps.d_member, self.info.d_pos, self.info.d_vel, self.info.d_force, 
												 self.xi, self.info.dt)
		#print(self.info.d_box[0], self.info.d_box[1], self.info.d_box[2])
		cu_iso_particle_scale[nblocks, self.block_size](self.ci.ps.nme, self.ci.ps.d_member, self.info.d_pos, self.info.d_image, self.info.d_box, self.pi)
		self.info.box = self.info.d_box.copy_to_host()
		#cu_iso_box_scale[1,32](self.info.d_box, self.pi)
		#print(self.info.box, self.ci.pressure, self.ci.temp)
		# self.info.pos = self.info.d_pos.copy_to_host()
		# for i in range(0, self.info.pos.shape[0]):
			# print(i, self.info.pos[i][0], self.info.pos[i][1],  self.info.pos[i][2])
		# self.info.vel = self.info.d_vel.copy_to_host()
		# for i in range(0, self.info.vel.shape[0]):
			# print(i, self.info.vel[i][0], self.info.vel[i][1],  self.info.vel[i][2])			
	def secondstep(self, timestep):
		self.ci.calculate(timestep)
		curr_T = self.ci.temp
		curr_P = self.ci.pressure
		if(curr_T<1.0e-6):
			curr_T=0.0001
		self.xi = (1.0+self.info.dt*(self.temp/curr_T-1.0)/self.tau)**(1/2) 
		self.pi = (1 + self.info.dt*(curr_P - self.pres)/self.taup)**(1/3)
		#if timestep % 100 == 0:
		#	print(self.ci.temp)
		nblocks = math.ceil(self.ci.ps.nme / self.block_size)
		cu_second_step[nblocks, self.block_size](self.ci.ps.nme, self.ci.ps.d_member, self.info.d_vel, self.info.d_force, self.xi, self.info.dt)
		# self.info.pos = self.info.d_pos.copy_to_host()
		# for i in range(0, self.info.pos.shape[0]):
			# print(i, self.info.pos[i][0], self.info.pos[i][1],  self.info.pos[i][2])
		# self.info.vel = self.info.d_vel.copy_to_host()
		# for i in range(0, self.info.vel.shape[0]):
			# print(i, self.info.vel[i][0], self.info.vel[i][1],  self.info.vel[i][2])			
		
	def register(self, timestep):
		self.info.compute_properties['temperature']=True		
		self.info.compute_properties['pressure']=True		
		self.info.compute_properties['potential']=True		
		self.info.compute_properties['momentum']=True		
