%
% Example script: find the nearest sparse matrix to 
% Matrix Market's benchmark matrix orani678
% that has an eigenvalue with |lambda|^2 <= 1.7293467e-10.
%
% This should give the same value as the last row of the table in 
% [Guglielmi-Lubich-Sicilia, Sec. 7.1].

addpath(genpath('manopt/manopt'));

A = mmread('orani678.mtx');

problem = nearest_unstable_sparse([], @(x) inside_disc(x, sqrt(1.7293467e-10)), A);

options = struct();

% options.maxiter = 10000;
% options.solver = @rlbfgs;
% options.verbosity = 1;

options.maxiter = 1000;
options.solver = @trustregions;
options.verbosity = 1;
options.epsilon_decrease = 0.8;

% specifies an initial value for y in the augmented Lagrangian method. 
% If isempty(options.y), the vanilla penalty method is used.
options.y = zeros(size(A,1), 1);

options.outer_iterations = 70;

[x cost info] = penalty_method(problem, [], options);
x_reg = problem.recover_exact(x, 1/eps); 
cost_reg = problem.cost(x_reg, struct());
fprintf('Optimal cost function after inserting zeros in A*v: %e. sqrt(cost) = %e\n', cost_reg, sqrt(cost_reg));
