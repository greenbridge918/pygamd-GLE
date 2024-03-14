# pygamd-GLE
A python packages that can be used for classical generalized Langevin equation (GLE) simulations.
More specific, the GLE integrator adopts the form of numerical integration rather than extend Markovian method to handle 
position-undependent homogeneous memory kernel. We develop a method to effectively split a long-lived memory kernel to two parts, a delta function 
and short-time memory kernel, to deal with the system with very slow relaxation, e.g., linear polymer. More details can be found in [https://doi.org/10.1021/jacsau.3c00756](https://doi.org/10.1021/jacsau.3c00756).

We also provide interested readers with the CG model from article, as well as scripts about the iterative procedure.

## Change Statement
This project is a modified version of open-source pygamd-v1 packages. Any questions about installation, citation and documentation can refered to [https://github.com/youliangzhu/pygamd-v1](https://github.com/youliangzhu/pygamd-v1).
It should be noted that you need to install SciPy packages if you want to use numerical potentials (table potentials) in MD simulations.

The following were modified or added:

1. pygamd/integrations/gle.py added
2. pygamd/integrations/BerendsenNvt.py added
3. pygamd/integrations/BerendsenNpt.py added
4. pygamd/forces/table_reader added
5. pygamd/forces/pair.py modified
6. pygamd/forces/potentials/bonded_library.py modified
7. pygamd/forces/potentials/nb_library.py modified
