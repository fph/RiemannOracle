%
% Polynomial GCD example
%

p = [1; 2; 3; -4; 5; -6; 7];
q = [8; 9; -10; 11; -12];
d = 3;

degp = length(p) - 1;
degq = length(q) - 1;

A = [1/sqrt(degq) * polytoep(p, degq-1)   1/sqrt(degp) * polytoep(q, degp-1)];

% Achtung: the following method to construct the basis works only if 
% all coefficients in p and q are distinct
P = autobasis(A);

options = struct();
options.y = 0;
options.verbosity = 0;
options.maxiter = 100;
options.max_outer_iterations = 40;

problem = nearest_nullity_structured_dense(P, A, d, true);
[x, cost, info, results] = penalty_method(problem, [], options);

Apert = A + info.Delta;
pp = sqrt(degq) * Apert(1:length(p), 1);
qq = sqrt(degp) * Apert(1:length(q), length(q));

format short e

fprintf('coefficients of p:\n')
p
fprintf('coefficients of q:\n')
q
fprintf('coefficients of pp:\n')
pp
fprintf('coefficients of qq:\n')
qq
fprintf('norm([p-pp; q-qq]) = %e\n', norm([p-pp; q-qq]));
fprintf('svd of resultant(pp,qq):\n')
svd(Apert).'
fprintf('roots of pp:\n');
r1 = roots(pp).'
fprintf('roots of qq:\n');
r2 = roots(qq).'
fprintf('At least %d of them should coincide.\n', d);
