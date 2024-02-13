%
% Nearest singular matrix to grcar(8) with the same structure
% 
% This is a tricky case since M is tall-thin

n = 8;
A = gallery('grcar', n);
P = [];
P(:,:,1) = diag(ones(n-1,1), -1);
P(:,:,2) = diag(ones(n,1), 0);
P(:,:,3) = diag(ones(n-1,1), 1);
P(:,:,4) = diag(ones(n-2,1), 2);
P(:,:,5) = diag(ones(n-3,1), 3);
% TODO: check
% P(:,:,6) = diag(ones(n-4,1), 4);
% P(:,:,7) = diag(ones(n-5,1), 5);
% P(:,:,8) = diag(ones(n-6,1), 6);

for k = 1:size(P,3)
    P(:,:,k) = P(:,:,k) / norm(P(:,:,k), 'fro');
end

problem = nearest_singular_structured_dense(P, A, true);

% regproblem = apply_regularization(problem, 0.2, randn(size(A,1),1));
% checkgradient(regproblem)
% checkhessian(regproblem)

options.outer_iterations = 30;
[x cost info] = penalty_method(problem, [], options);
regproblem = apply_regularization(problem, info.last_epsilon, info.y);
[Delta store] = regproblem.minimizer(x, struct());

fprintf('Cost: %e, sigma_min: %e\n', cost, min(svd(A+Delta)));