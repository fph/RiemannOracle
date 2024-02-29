This is work-in-progress code for the solution of certain matrix nearness problems such as: what is the nearest matrix to a given matrix that has an eigenvalue in a certain closed region ? I.e., minimize 

$$
  f(X) = \min_{X \colon \Lambda(X)\cap \mathbb{\Omega}\neq\emptyset} \\| A-X \\| _F. 
$$

The problem is reduced to optimization on manifolds and solved using the package [Manopt](https://www.manopt.org/). 

## How to use

You will need a [downloaded copy of Manopt](https://www.manopt.org/downloads.html) in `./manopt` or in the Matlab path.

See the files `src/example*.m` for usage examples.
