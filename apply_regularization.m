function problem = apply_regularization(problem, epsilon, y)
% generate a new Manopt problem structure with regularization
%
% Create a new problem structure in which cost(), egrad(), ehess(),
% minimizer() include the specified regularization, by forwarding these
% calls to gencost(), genegrad(), etc.
%
% Note that only the two-argument versions of the functions are used, for
% ease of writing, so if you want to call one of these functions outside the
% solver you need to pass an empty struct: e.g.,
%
% >> problem.cost(v, struct())
%

problem.cost = @(v, store) problem.gencost(epsilon, y, v, store);
problem.egrad = @(v, store) problem.genegrad(epsilon, y, v, store);
if isfield(problem, 'genehess')
    problem.ehess = @(v, w, store) problem.genehess(epsilon, y, v, w, store);
end

problem.minimizer = @(v, store) problem.genminimizer(epsilon, y, v, store);
problem.constraint = @(v, store) problem.genconstraint(epsilon, y, v, store);
