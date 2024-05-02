% An example for the specialised polynomial algorithm
rng(1)

% m is the size of the matrix polynomial
m = 8;
% k is the degree of the matrix polynomial
k = 2;
% d is the largest allowed minimal index
d = floor((k*(m-1))/2);

A0 = randn(m) + 1i*randn(m);
A1 = randn(m) + 1i*randn(m);
A2 = randn(m) + 1i*randn(m);

% Initial guess for the vector in the kernel
V0 = randn(m,d+1);
V0 = V0./norm(V0,'f');

% Parameters and options used by the algorithm
options = struct();
options.maxiter = round(sqrt(k)*40);
options.tolgradnorm = 1e-6;
options.solver = @trustregions;
options.verbosity = 1;
options.epsilon_decrease = 'f';
options.max_outer_iterations = 800;
options.stopping_criterion = 1e-14;

use_hessian = true;


% Right kernel:
A = [A0 A1 A2];

problem = nearest_singular_polynomial(A, d, use_hessian);
          
[V_right,~,info_right, results_right] = penalty_method(problem, V0, options);

% Left kernel:
A = [A0.' A1.' A2.'];

problem = nearest_singular_polynomial(A, d, use_hessian);
          
[V_left,~,info_left, results_left] = penalty_method(problem, V0, options);

% Choose the smaller one
norm(info_left.Delta,'fro')
norm(info_right.Delta,'fro')
