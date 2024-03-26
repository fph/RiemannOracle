% An example for the specialised polynomial algorithm
clear all;

rng(5)
m = 8;
A0 = randn(m) + 1i*randn(m);
A1 = randn(m) + 1i*randn(m);
A2 = randn(m) + 1i*randn(m);

n = length(A1);

V0 = randn(n,floor((2*(n-1))/2)+1);
V0 = V0./norm(V0,'f');

d = size(V0, 2) - 1;

A = [A0 A1 A2];

options = struct();
options.maxiter = 5000;
options.maxtime = 4;
options.tolgradnorm = 1e-6;
% options.debug=0;
options.solver = @trustregions;
options.verbosity = 1;
% options.epsilon_decrease = 'f';
options.max_outer_iterations = 10;
options.y = zeros(n,d+3);


use_hessian = true;


% Right kernel:
problem = nearest_singular_polynomial(A, [], use_hessian);
          
[V_right,~,info_right] = penalty_method(problem, V0, options);

% Left kernel:
A = [A0.' A1.' A2.'];

problem = nearest_singular_polynomial(A, [], use_hessian);
          
[V_left,~,info_left] = penalty_method(problem, V0, options);

% Choose the smaller one
norm(info_left.Delta,'fro')
norm(info_right.Delta,'fro')

