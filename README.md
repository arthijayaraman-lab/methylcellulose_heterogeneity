Associated code for **citation**. A higher level description of the methods can be found in that paper. If you use any of the code or data in this repo, please cite the paper. While the trajectories are too large to share here, we have done our best to include text/csv files of all key results from simulations.

The repository is broken up into the following sections:
- bonded_bi: Boltzmann inversion from atomistic simulation data to provide bonded parameters for coarse-grained model
- nonbond_bo: Bayesian optimization of nonbonded energy parameters used in the coarse-grained model
- heterogeneities: codes for generating monomer sequences based on sequence and compositional heterogeneity
- solgel_boundaries: Broad initial sampling to determine solgel boundaries for different methylcellulose designs
- fibril_structure: Large scale MD simulations and analysis for determining fibril dimensions
- util: miscellaneous functions

The python environment used for this work is included in mc-env.yml, and can be installed using the following: 
```
conda env create -f mc-env.yml
```


