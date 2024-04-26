function [Apert, nrm, nullity] = grcar_nullity(n, d, structure)
% Nearest singular matrix to grcar(n)
% nullity = d > 1
% structure can be 'sparsity' or 'toeplitz'
% 
% This is a tricky case since M is tall-thin and not all vectors v
% are feasible.
rng(2)


%grcar matrix of dimension n
A = gallery('grcar', n);

if d > n
   error('the prescribed nullity should be d < n')
end

if (d == 0) || (n == 0)
   error('add a nullity or a dimension')
end

switch structure
    case 'toeplitz'
        a = randn;
        c = [a ; randn(n-1,1)];
        r = [a; randn(n-1,1)];
        B = toeplitz(c,r);
    case 'sparsity'
        B = randn(n);
        B = tril(triu(B,-1),3);
end

P = autobasis(B);

problem = nearest_nullity_structured_dense(P, A, d, true);
%problem = nearest_singular_structured_dense(P, A, true);

options = struct();
options.max_outer_iterations = 30;
options.verbosity = 0;
% options.epsilon_decrease = 'f';
options.y = 0;

[x, cost, info, results] = penalty_method(problem, [], options);

Apert = A + info.Delta;

% check the computed nullity
s = svd(Apert);
nullity = sum(s < max(10*s(end), 1e-13 * s(1)));

nrm = norm(info.Delta,'fro');

% regproblem = apply_regularization(problem, info.last_epsilon, info.y);
% [Delta, AplusDelta, store] = regproblem.minimizer(x, struct());

fprintf('Cost: %e, sigma_min: %e\n', cost, min(svd(Apert)));
fprintf('Computed nullity: %d, Norm of the perturbation: %e vs Norm of A: %e \n', nullity, norm(info.Delta,'fro'), norm(A,'fro'))