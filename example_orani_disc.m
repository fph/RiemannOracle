addpath(genpath('manopt/manopt'));

A = mmread('orani678.mtx');

problem = nearest_unstable_sparse([], @(x) inside_disc(x, sqrt(1.7293467e-10)), A);

options = struct();
options.maxiter = 20000;
options.verbosity = 1;
options.solver = @trustregions;
% specifies an initial value for y in the augmented Lagrangian method. 
% If isempty(options.y), the vanilla penalty method is used.
options.y = 0;
options.outer_iterations = 20;

x = penalty_method(problem, [], options);
x_reg = problem.recover_exact(x, 1/eps); 
cost_reg = problem.cost(x_reg, struct());
fprintf('Optimal cost function after inserting zeros in A*v: %e.\n', cost_reg);