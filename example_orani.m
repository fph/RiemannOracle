addpath(genpath('manopt/manopt'));

A = mmread('orani678.mtx');

problem = nearest_singular([], A, true);

options = struct();
options.maxiter = 10000;
options.verbosity = 1;
options.solver = @trustregions;
% specifies an initial value for y in the augmented Lagrangian method. 
% If isempty(options.y), the vanilla penalty method is used.
options.y = 0; 

penalty_method(problem, [], options);
