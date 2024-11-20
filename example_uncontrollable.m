% An example for the nearest uncontrollable matrix polynomial
rng(1)

% the size of the matrix polynomial is mxn
m = 30;
n = 10;
% k-1 is the degree of the matrix polynomial
k = 3;

% Unstructured matrix
A0 = randn(m,n) + 1i*randn(m,n);
A1 = randn(m,n) + 1i*randn(m,n);
A2 = randn(m,n) + 1i*randn(m,n);

% Some oddly structured example
% % A0 = randn(m,n); A0(1:n,1:n) = A0'*A0;
% % A1 = randn(m,n); A1(1:n,1:n) = A1'*A1;
% % A2 = randn(m,n); A2(1:n,1:n) = A2'*A2;

A = reshape([A2 A1 A0],m,n,3);

P = autobasis([A2 A1 A0]);

% Parameters and options used by the algorithm
options = struct();
options.maxiter = round(sqrt(k)*40);
options.tolgradnorm = 1e-6;
options.solver = @trustregions;
options.verbosity = 2;
options.epsilon_decrease = 'f';
options.max_outer_iterations = 800;
options.stopping_criterion = 1e-14;

% Uncomment for augmented Lagrangian 
% options.y=0;

use_hessian = false;
warning('off', 'manopt:getHessian:approx');

problem = nearest_uncontrollable_structured_dense(P, A, use_hessian);
          
[V,~,info, results] = penalty_method(problem, [], options);

Delta_norm = norm(info.Delta,'fro');
disp(['Distance to uncontrollability: ', num2str(Delta_norm)])

%% Sanity check for the constraint
% wI = kron(V.X.^(k-1:-1:0).',V.Y);
% A2 = reshape(A,m,n*k);
% norm((A2 + info.Delta)*wI)

