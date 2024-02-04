function problem = apply_regularization(problem, epsilon, y)
% use the problem.gen* function to generate a modified problem w/
% regularization
%
% we go for this approach because we'd like to keep the "everything is a
% struct" approach from Manopt

problem.cost = @(v, store) problem.gencost(v, epsilon, y, store);
problem.egrad = @(v, store) problem.genegrad(v, epsilon, y, store);
if isfield(problem, 'ehess')
    problem.ehess = @(v, w, store) problem.genehess(v, w, epsilon, y, store);
end

problem.minimizer = @(v, store) problem.genminimizer(v, epsilon, y, store);

