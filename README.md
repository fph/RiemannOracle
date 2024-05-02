## Riemannian-Oracle
This repository contains the code for the solution of certain nearness problems in matrix theory, following the idea in 

M. Gnazzo, V. Noferini, L. Nyman, F. Poloni. "Riemann-Oracle: A general-purpose Riemannian optimizer to solve nearness problems in matrix theory", available on arXiv, 2024.

We consider nearness problem in the general form: given $A$ without the property $\mathfrak{P}$, minimize the 

$$
  f(\Delta) = \min \left\lbrace \\| A-X \\| _F : X \mbox{with} \mathfrak{P} \right\rbrace. 
$$

The problem is reduced to optimization on manifolds and solved using the package [Manopt](https://www.manopt.org/). 

## How to use

You will need a [downloaded copy of Manopt](https://www.manopt.org/downloads.html) in `./manopt` or in the Matlab path.

See the files `src/example*.m` for usage examples.
