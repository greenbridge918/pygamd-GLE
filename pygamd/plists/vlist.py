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
import pygamd.snapshots.box
import numpy as np
class vlist:
	
	#定义构造方法
	def __init__(self, info, rcut):
		self.rcut = rcut
#		print(self.rcut)
		self.info=info
		self.pos=info.pos
#		print(self.pos)
		self.npa = info.npa
#		print(self.np)
		self.list = [[] for i in range(self.npa)]
	def calculate(self):
		for i in range(0, self.npa):
			for j in range(i+1, self.npa):
				dp = self.pos[i] - self.pos[j]
				pygamd.snapshots.box.box_min_dis(dp, self.info.box)
				rsq = (dp*dp).sum()
				r = math.sqrt(rsq)
				if r < self.rcut:
					self.list[i].append(j)
					self.list[j].append(i)
	def speak(self):
		print(self.list)
