function problem = apply_regularization(problem, epsilon, y, force_hessian)
% use the problem.gen* function to generate a modified problem w/
% regularization
%
% we go for this approach because we'd like to keep the "everything is a
% struct" approach from Manopt

if not(exist('force_hessian', 'var'))
    force_hessian = false;
end

problem.cost = @(v, store) problem.gencost(epsilon, y, v, store);
problem.egrad = @(v, store) problem.genegrad(epsilon, y, v, store);
if isfield(problem, 'ehess') || force_hessian
    problem.ehess = @(v, w, store) problem.genehess(epsilon, y, v, w, store);
end

problem.minimizer = @(v, store) problem.genminimizer(epsilon, y, v, store);
problem.constraint = @(v, store) problem.genconstraint(epsilon, y, v, store);

