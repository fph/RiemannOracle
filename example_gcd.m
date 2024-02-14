%
% Polynomial GCD example
%

p = [1 2 3 -4 5 -6 7];
q = [8 9 -10 11 -12];
d = 3;

degp = length(p) - 1;
degq = length(q) - 1;

A = [1/sqrt(degq) * polytoep(p, degq-1).';
     1/sqrt(degp) * polytoep(q, degp-1).'];

% Achtung: the following method to construct the basis works only if 
% all coefficients in p and q are distinct
P = autobasis(A);

problem = nearest_nullity_structured_dense(P, A, d, true);
[x cost info] = penalty_method(problem);

Apert = A + info.Delta;
pp = sqrt(degq) * Apert(1, 1:length(p));
qq = sqrt(degp) * Apert(length(q), 1:length(q));

fprintf('coefficients of p:\n')
p
fprintf('coefficients of q:\n')
q
fprintf('coefficients of pp:\n')
pp
fprintf('coefficients of qq:\n')
qq
fprintf('norm(p-pp)^2 + norm(q-qq)^2 = %e\n', norm(p-pp)^2 + norm(q-qq)^2);
fprintf('svd of resultant(pp,qq):\n')
svd(Apert).'
fprintf('roots of pp:\n');
r1 = roots(pp).'
fprintf('roots of qq:\n');
r2 = roots(qq).'
fprintf('At least %d of them should coincide.\n', d);
