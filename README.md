## Riemannian-Oracle
This repository contains the code for the solution of certain nearness problems in matrix theory, following the idea in 

M. Gnazzo, V. Noferini, L. Nyman, F. Poloni. "Riemann-Oracle: A general-purpose Riemannian optimizer to solve nearness problems in matrix theory", soon available on arXiv, 2024.

We consider nearness problem in the general form: given $A$ without the property $\mathfrak{P}$, minimize the 

$$
  f(\Delta) = \min \left\lbrace \\| A-\Delta \\| _F : \Delta \mbox{ with } \mathfrak{P}, \Delta \in \mathcal{S}\right\rbrace, 
$$

with $\mathcal{S}$ linear subspace. The problem is reduced to optimization on manifolds and solved using the package [Manopt](https://www.manopt.org/). 

## Main features
The code is able to tackle different matrix nearness problems, such as:
* **Nearest structured singular matrix**: check the code in <code>nearest_singular_structured_dense.m</code> for the general framework;
* **Nearest singular matrix polynomial**: check the code in <code>nearest_singular_polynomial.m</code> for the general framework;
* **Nearest structured unstable matrix**: check the code in <code>nearest_unstable_structured_dense.m</code> for the general framework;
* **Nearest matrix with prescribed nullity**: check the code in <code>nearest_nullity_structured_dense.m</code> for the general framework;
* **Approximate GCD of prescribed degree betweeen scalar polynomials**: check the code in <code>example_zeng2.m</code> for an illustrative example.

The function <code>penalty_method.m</code> contains the optimization procedure, using the Riemannian-Oracle method. It can be used to address other nearness problems, besides the one mentioned above.

## How to use

You will need a [downloaded copy of Manopt](https://www.manopt.org/downloads.html) in `./manopt` or in the Matlab path.

See the files `src/example*.m` for usage examples.
