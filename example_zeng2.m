%
% Test 2 in Zeng, "The Numerical Greatest Common Divisor of Univariate
% Polynomials"
%

% sought gcd degree
d = input('Degree of sought gcd? ');

% create polynomials from the example
p = 1;
q = 1;
for j = 1:10
    xj = (-1)^j * j/2;
    p = conv(p, [1; -xj]);
    q = conv(q, [1; -xj+10^(-j)]);
end

% normalize as in Bini-Boito
p = p / norm(p);
q = q / norm(q);

degp = length(p) - 1;
degq = length(q) - 1;

% generate "fake" polynomials with the same degrees 
% and (distinct) integer coefficients
% to produce a perturbation basis with autobasis.
% This is faster to write than constructing the basis by hand.
fake_p = 1 : degp+1; 
fake_q = 2*degp+1 : 2*degp+degq+1;
fake_A = [polytoep(p, degq-d) ...
     polytoep(q, degp-d)];
P = autobasis(fake_A);

% now the real matrix A, scaled so that
% norm(Delta)_F^2 = norm(p-deltap)^2 + norm(q-deltaq)^2
A = [1/sqrt(degq-d+1) * polytoep(p, degq-d) ...
    1/sqrt(degp-d+1) * polytoep(q, degp-d)];

options = struct();
options.y = 0;
options.maxiter = 1000;
options.verbose = 1;
options.outer_iterations = 40;

problem = nearest_singular_structured_dense(P, A, true);
[x cost info] = penalty_method(problem, [], options);

Apert = A + info.Delta;
pp = sqrt(degq-d+1) * Apert(1:degp+1, 1);
qq = sqrt(degp-d+1) * Apert(1:degq+1, degq-d+2);

uu = x(1:degq-d+1);
vv = x(degq-d+2:end);
gg = deconv(pp, vv);  % reconstruct gcd TODO: improve

nearness = norm([conv(gg,vv)-p conv(gg,-uu)-q])