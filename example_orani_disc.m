%
% Example script: find the nearest sparse matrix to 
% Matrix Market's benchmark matrix orani678
% that has an eigenvalue with |lambda|^2 <= r.
%
% This should return values comparable to those in [Guglielmi-Lubich-Sicilia, Table 7.2].
%
% Run as it is, the script takes quite long to run, since there are many examples,
% they are large-scale, and the exact Hessian is not available.

A = mmread('orani678.mtx');

squaredradii = [
    1.1019564e-2;
    9.5284061e-4;
    2.5263758e-4;
    6.5050153e-5;
    1.6503282e-6;
    4.1561289e-6;
    1.0428313e-6;
    2.6118300e-7;
    6.5355110e-8;
    9.6346192e-9;
    1.7293467e-10;
    ];

datatable = table();
for k = 1:length(squaredradii)
    
    problem = nearest_unstable_sparse([], @(x) inside_disc(x, sqrt(phieps)), A);
    
    options = struct();
    
    % options.maxiter = 10000;
    % options.solver = @rlbfgs;
    % rlbfgs seems to work better when the exact Hessian is not available
    
    % options.maxiter = 1000;
    % options.solver = @trustregions;

    options.tolgradnorm = 1e-10;
    options.verbosity = 1;
    options.epsilon_decrease = 0.7;
    options.max_outer_iterations = 40;
    
    % specifies an initial value for y in the augmented Lagrangian method. 
    % If isempty(options.y), the vanilla penalty method is used.
    options.y = zeros(size(A,1), 1);
    
    [x, cost, info, results] = penalty_method(problem, [], options);
    x_reg = problem.recover_exact(x, 1/eps); 
    cost_reg = problem.cost(x_reg, struct());
    fprintf('Optimal value of f_{eps} with eps=%e: %e.\n', info.last_epsilon, cost);
    fprintf('Optimal value of f after inserting zeros in A*v: %e. sqrt(cost) = %e\n', cost_reg, sqrt(cost_reg));
    [Delta, lambda, store] = problem.minimizer(x_reg, struct());
    datatable.epsilon(k) = norm(Delta, 'fro');
    datatable.phiepsilon(k) = min(abs(eig(full(A+Delta))))^2;
    datatable
end
