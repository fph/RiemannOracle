function problem = apply_regularization(problem, epsilon, y, force_hessian)
% use the problem.gen* function to generate a modified problem w/
% regularization
%
% we go for this approach because we'd like to keep the "everything is a
% struct" approach from Manopt

if not(exist('force_hessian', 'var'))
    force_hessian = false;
end

problem.cost = @(v, store) problem.gencost(v, epsilon, y, store);
problem.egrad = @(v, store) problem.genegrad(v, epsilon, y, store);
if isfield(problem, 'ehess') || force_hessian
    problem.ehess = @(v, w, store) problem.genehess(v, w, epsilon, y, store);
end

problem.minimizer = @(v, store) problem.genminimizer(v, epsilon, y, store);
problem.constraint = @(v, store) problem.genconstraint(v, epsilon, y, store);

