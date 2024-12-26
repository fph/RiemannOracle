p = [1 2 6 9];
d = length(p) - 1;
A = compan(p);
P = autobasis(compan([1 -(1:d)]) - compan([1 zeros(1,d)]));

problem = nearest_defective_structured_dense(P, A);

options = struct();
options.y = 0;
options.tolgradnorm = 1e-10;
options.minstepsize = 1e-100;
options.max_outer_iterations = 20;
options.solver = @trustregions;
options.maxiter = 100;

[x cost info] = penalty_method(problem, [], options);

eig(A+info.Delta)
