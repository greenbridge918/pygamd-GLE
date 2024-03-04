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

import math
from numba import cuda
import numba as nb

# device functions
def cu_nonbonded(force_name):
	@cuda.jit(device=True)
	def _lj(rsq, param, fp):
		lj1 = param[0]
		lj2 = param[1]
		rcutsq = param[2]
		if rsq<rcutsq:
			r2inv = nb.float32(1.0)/rsq
			r6inv = r2inv * r2inv * r2inv
			f = r2inv * r6inv * (nb.float32(12.0) * lj1  * r6inv - nb.float32(6.0) * lj2)	
			p = r6inv * (lj1 * r6inv - lj2)
			fp[0]=f
			fp[1]=p
			
	@cuda.jit(device=True)
	def _harmonic(rsq, param, fp):
		alpha = param[0]
		r_cutINV = param[1]
		rcutsq = param[2]
		if rsq<rcutsq:
			rinv = nb.float32(1.0)/math.sqrt(rsq)
			omega = rinv - r_cutINV
			f = alpha*omega
			p = nb.float32(0.5)*alpha*omega*omega*rsq
			fp[0]=f
			fp[1]=p

	@cuda.jit(device=True)
	def _lj_coulomb(rsq, qi, qj, param, fp):
		lj1 = param[0]
		lj2 = param[1]
		coulomb_eff = param[2]
		rcutsq = param[3]
		if rsq<rcutsq:
			r2inv = nb.float32(1.0)/rsq
			rinv = math.sqrt(r2inv)
			r6inv = r2inv * r2inv * r2inv
			f = r2inv * r6inv * (nb.float32(12.0) * lj1  * r6inv - nb.float32(6.0) * lj2) + coulomb_eff*qi*qj*r2inv*rinv
			p = r6inv * (lj1 * r6inv - lj2) + coulomb_eff*qi*qj*rinv
			fp[0]=f
			fp[1]=p

	@cuda.jit(device=True)
	def _pair_table(rsq, param, fp):
	    #rcut = param[0]
		rcut = 1.75
		n_points = param[1]
		binsize = rcut / (n_points-1)
		n_coeff = n_points - 1
		p_spline_para = param[2:int(2+n_coeff*4)]
		f_spline_para = param[int(2+n_coeff*4):int(2+n_coeff*7)]
		table_points = param[int(2+n_coeff*7):]
		if rsq<rcut*rcut:
			r = rsq**(1/2)
			i = int(math.floor(r / binsize))
			p_C1, p_C2, p_C3, p_C4 = p_spline_para[int(i*4)], p_spline_para[int(i*4+1)], p_spline_para[int(i*4+2)], p_spline_para[int(i*4+3)]
			f_C1, f_C2, f_C3 = f_spline_para[int(i*3)], f_spline_para[int(i*3+1)], f_spline_para[int(i*3+2)]
			p = p_C1 * (r - table_points[i]) ** 3 + p_C2 * (r - table_points[i]) ** 2 + p_C3 * (r - table_points[i]) + p_C4
			f = - (f_C1 * (r - table_points[i]) ** 2 + f_C2 * (r - table_points[i]) + f_C3) / r
			fp[0]=f
			fp[1]=p	

	if force_name=="lj":
		return _lj
	elif force_name=="harmonic":
		return _harmonic
	elif force_name=="lj_coulomb":
		return _lj_coulomb
	elif force_name=="pair_table":
		return _pair_table

# host functions
def nonbonded(force_name):
	def _lj(rsq, param, fp):
		lj1 = param[0]
		lj2 = param[1]
		rcutsq = param[2]
		if rsq<rcutsq:
			r2inv = 1.0/rsq;
			r6inv = r2inv * r2inv * r2inv;
			f = r2inv * r6inv * (12.0 * lj1  * r6inv - 6.0 * lj2)	
			p = r6inv * (lj1 * r6inv - lj2)
			fp[0]=f
			fp[1]=p

	def _harmonic(rsq, param, fp):
		alpha = param[0]
		r_cutINV = param[1]
		rcutsq = param[2]
		if rsq<rcutsq:
			rinv = 1.0/math.sqrt(rsq)
			omega = rinv - r_cutINV
			f = alpha*omega
			p = 0.5*alpha*omega*omega*rsq
			fp[0]=f
			fp[1]=p

	def _lj_coulomb(rsq, qi, qj, param, fp):
		lj1 = param[0]
		lj2 = param[1]
		coulomb_eff = param[2]
		rcutsq = param[3]
		if rsq<rcutsq:
			r2inv = 1.0/rsq
			rinv = math.sqrt(r2inv)
			r6inv = r2inv * r2inv * r2inv
			f = r2inv * r6inv * (12.0 * lj1  * r6inv - 6.0 * lj2) + coulomb_eff*qi*qj*r2inv*rinv
			p = r6inv * (lj1 * r6inv - lj2) + coulomb_eff*qi*qj*rinv
			fp[0]=f
			fp[1]=p

	def _pair_table(rsq, param, fp):
	    rcut = param[0]
	    n_points = param[1]
	    binsize = rcut / (n_points-1)
	    n_coeff = n_points - 1
	    p_spline_para = param[2:int(2+n_coeff*4)]
	    f_spline_para = param[int(2+n_coeff*4):int(2+n_coeff*7)]
	    table_points = param[int(2+n_coeff*7):]
	    if rsq<rcut*rcut:
	        r = rsq**(1/2)
	        i = int(math.floor(r / binsize))
	        p_C1, p_C2, p_C3, p_C4 = p_spline_para[int(i*4)], p_spline_para[int(i*4+1)], p_spline_para[int(i*4+2)], p_spline_para[int(i*4+3)]
	        f_C1, f_C2, f_C3 = f_spline_para[int(i*3)], f_spline_para[int(i*3+1)], f_spline_para[int(i*3+2)]
	        p = p_C1 * (r - table_points[i]) ** 3 + p_C2 * (r - table_points[i]) ** 2 + p_C3 * (r - table_points[i]) + p_C4
	        f = - (f_C1 * (r - table_points[i]) ** 2 + f_C2 * (r - table_points[i]) + f_C3) / r
	        fp[0]=f
	        fp[1]=p

	if force_name=="lj":
		return _lj
	elif force_name=="harmonic":
		return _harmonic
	elif force_name=="lj_coulomb":
		return _lj_coulomb
	elif force_name=="pair_table":
		return _pair_table

		
