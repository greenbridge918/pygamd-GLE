"""
This script is used for iterative development of numerical memory kernel by 
Iterative Memory Reconstruction VACF (IMRV) method. It should be in the same
directory with `template`.

Required files in `template`: *.table (conservative interactions with numerical form), 
coor.mst (initial configuration), md.py (md script), cal_vacf.py (calculate VACF) and 
AA_VACF.txt (reference VACF)

example:
python IMRV.py
"""
import numpy as np
import sys
import os
import subprocess
from scipy import signal
from scipy import interpolate


def WeightFunc(itn, n_itgs, dt, tloc):
    wf = np.zeros((n_itgs), dtype=np.float32)
    for i in range(n_itgs):
        r = i*dt/tloc
        if r <= itn/2:
            wf[i] = 1
        elif (r < i/2+1) and (r > i/2):
            wf[i] = 1 - r + i/2
        else:
            wf[i] = 0
    return wf
    """ A time window function used for achieving a well-behaved
    convergence. The range of correction applied on the 
    memory kernel increases step by step.

    Parameter:
        itn: Iteration number / int
        n_itgs: The number of numerical points of memory kernel / int
        dt: Timestep / float
        tloc: A `correction` time that localizes the
        time window in which the correction is applied / int
    Return:
        A numpy array with the size (n_itg).

    """

def MappingFunc(vacf, n_itgs, dt, mass, temperature, order):
    X = np.arange(n_itgs)*dt
    cubic = interpolate.CubicSpline(X, vacf).derivative(1)
    one_order_mf = -mass**2*cubic(X)/temperature
    zero_order_mf = -mass**2*vacf/temperature
    if order == 1:
        mf = one_order_mf
    elif order == 0:
        mf = zero_order_mf
    else:
        raise ValueError('Invaild order.')
    return mf
    """ The correction is the difference between mapping functions 
    calculated from AA and GLE simulation at each iterations.

    Parameter:
        vacf: Velocity autocorrelation. / numpy array
        n_itgs: The number of numerical points of memory kernel. / int
        dt: Timestep. / float
        mass: Mass of CG bead. / float
        temperature: Thermodynamic temperature, i.e., k_B T. /float
        order: The order of mapping function determines whether the time
        series used the first-order derivative of the VACF or itself. / int
    Return:
        A numpy array with the size (n_itg).

    """

class IMRV(object):
    """Iterative procedure

    Attribution:
        target: Iterative target, i.e., VACF. / numpy array
        t: Time series of memory kernel / numpy array
        n_itgs: The number of numerical points of memory kernel. / int
        temperature: Thermodynamic temperature, i.e., k_B T. / float
        mass: Mass of CG bead. / float
        dt: Timestep. / float
        init: Initialized memory kernel, a delta function at 0th iteration,
        otherwise read from the previous iteration. / numpy array
        kernel: Current memory kernel at current iteration. / numpy array
        current: Current iteration number. / int
        max: Max iteration number. / int
        order: The order of mapping function. / int
        is_converged: Check convergence. / bool
        gpuid: Which GPU is used to run MD simulation. / int

    """
    def __init__(self, target_files, n_itgs, n_iteration, max_iteration, dt, mass, temperature, gamma, order, gpuid):
        """ target_files: the directory of target VACF.
            gamma: The integrated value of memory kernel. / float
        """

        target = np.loadtxt(f"{target_files}").T
        cubic = interpolate.CubicSpline(target[0], target[1])
        assert n_iteration >= 0
        self.t = np.arange(n_itgs) * dt
        self.target = cubic(self.t)
        self.n_itgs = n_itgs
        self.temperature = temperature
        self.mass = mass
        self.dt = dt
        self.gpuid = gpuid
        if n_iteration == 0 :
            self.init = self.GenerateInitKernel(n_itgs, dt, gamma)
        else:
            self.init = np.loadtxt(f"{n_iteration-1}th/AfterKernel.txt")[:,1]/np.sqrt(self.mass) # Unit conversion, as the reduced mass 1 = 104amu


        self.kernel = self.init
        self.current = n_iteration
        self.max = n_iteration + max_iteration
        self.order = order
        self.is_converged = False
    
    
    def BeginIteration(self):
        """ Begin the iterative procedure step by step"""
        for i in range(self.current, self.max):
            if self.is_converged == True:
                break
            # 1. Make directory, write memory kernel before correction and copy necessary files;
            # 2. Run MD simulation; 3. Calculate the VACF.
            if os.path.isdir(f"{i}th"):
                os.system(f"rm -rf {i}th")
            os.system(f"mkdir {i}th")
            np.savetxt(f"{i}th/BeforeKernel.txt", np.c_[self.t, self.kernel*np.sqrt(self.mass)])
            os.system(f"cd {i}th; cp ../template/*.table ../template/md.py ../template/coor.mst ../template/cal_vacf.py ../template/AA_VACF.txt ../template/plot.py .")
            p = subprocess.Popen(f"cd {i}th; python md.py {self.dt} --gpu={self.gpuid} >log 2>err", shell=True)
            return_code = p.wait()
            self.is_running(return_code, i, "MD")

            p = subprocess.Popen(f"cd {i}th; python cal_vacf.py {self.dt}", shell=True)
            return_code = p.wait()
            self.is_running(return_code, i, "Acf Calculation")

            cg_vacf = np.loadtxt(f"{i}th/VACF.txt")[:self.n_itgs,1]
            X = np.arange(self.n_itgs)*self.dt
            # 4. Add correction into memory kernel according to mapping functions
            metric_func = WeightFunc(i, self.n_itgs, self.dt, 10*self.dt)
            reduced_parameter = np.ones((self.n_itgs))
            reduced_parameter[:20] = 0.2 # it may enhance the numerical stability in iteration process by reducing the contribution from very short time
            deltak = MappingFunc(self.target, self.n_itgs, self.dt, self.mass, self.temperature, self.order) - MappingFunc(cg_vacf, self.n_itgs, self.dt, self.mass, self.temperature, self.order)
            deltak[0] = 0
            self.kernel += reduced_parameter*deltak*metric_func
            k0 = self.kernel[0]
            # 5. Smooth the memory kernel using savgol filter
            self.kernel = signal.savgol_filter(self.kernel[1:],9,3)
            # 6. Determine the K_0 value by difference between integrated value of memory kernel and summation of all t>0 points of current memory kernel
            self.kernel = np.concatenate(([k0], self.kernel))
            itg_val = (self.kernel[1:] * self.dt).sum()
            self.kernel[0] = (1780 - itg_val) * (2/self.dt)
            # 7. Save the memory kernel after iteration
            np.savetxt(f"{i}th/AfterKernel.txt", np.c_[self.t, self.kernel*np.sqrt(self.mass)])
            # Determine if the iteration converges
            if ((self.target - cg_vacf)**2).sum() < 0.0006:
                self.is_converged = True

    @staticmethod
    def GenerateInitKernel(n_itgs, dt, gamma):
        X = np.arange(n_itgs) * dt
        K = np.zeros((n_itgs))
        s = 2 * 1/dt # It is noted that 2 comes from the half contribution at t=0 (just image a split delta function)
        K[0] = s*gamma
        return K
    """ Generate initilized memory kernel with a delta function

    Parameter:
        n_itgs: The number of numerical points of memory kernel. / int
        dt: Timestep. / float
        gamma: The integrated value of memory kernel that 
        determined from a Langevin simulation by matching diffusion coefficient. / float
    Return:
        A numpy array with the size (n_itg).

    """

    @staticmethod
    def is_running(status, i, loc):
        if (status):
            sys.exit(f"{i}th iteration has failed, please check and retry iteration! (wrong location: {loc})")
    """ Check whether the iteration is working properly.
    """

imrv = IMRV("template/AA_VACF.txt", n_itgs=400, n_iteration=0, max_iteration=200, dt=0.005, mass=104, temperature=4.16, gamma=1780, order=0, gpuid=0)
imrv.BeginIteration()


            
            



        
