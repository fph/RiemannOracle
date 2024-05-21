%
% Boito 8.2.1 in "Toward the best algorithm for 
% approximate GCD of univariate polynomials", Kosaku Nagasaka,
% https://doi.org/10.1016/j.jsc.2019.08.004
%

% sought gcd degree
k = input('Degree of the exponent?')
d = input('Degree of sought gcd? ');

% this is a global multiplicative scaling of the problem that is sent to Manopt
% we add this additional parameter to play around with, since it seems that
% some of the inner tolerances are absolute and hence scaling-dependent.

scaling = 1;

% setting options
options = struct();
options.y = 0;
options.maxiter = 200;
options.verbosity = 1;
options.max_outer_iterations = 60;
options.epsilon_decrease = 0.5;
options.starting_epsilon = 1 * scaling;
options.tolgradnorm = 1e-12 * scaling;

x = sym('x');

p = expand((x^3 + 3*x - 1) * (x-1)^k);
q = diff(p, x);

p = transpose(double(coeffs(p, 'All')));
q = transpose(double(coeffs(q, 'All')));
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
fake_A = [polytoep(fake_p, degq-d) ...
     polytoep(fake_q, degp-d)];
P = autobasis(fake_A);

% now the real matrix A, scaled so that
% norm(Delta)_F^2 = norm(p-deltap)^2 + norm(q-deltaq)^2
A = [1/sqrt(degq-d+1) * polytoep(p, degq-d) ... % Sylvester matrix
    1/sqrt(degp-d+1) * polytoep(q, degp-d)];

alpha = [p;q];
problem = nearest_singular_structured_dense(P, scaling*alpha, true);
[x, cost, info, results] = penalty_method(problem, [], options);

Apert = A + 1/scaling * info.Delta;

s = svd(Apert);
nullity = sum(s < max(10*s(end), 1e-13 * s(1)));

pp = sqrt(degq-d+1) * Apert(1:degp+1, 1);
qq = sqrt(degp-d+1) * Apert(1:degq+1, degq-d+2);
uu0 = x(1:degq-d+1);
vv0 = x(degq-d+2:end);
x0 = x;

if nullity > 1
    warning('nullity > 1, we need to remove spurious factors from the cofactors.');
    d = d + nullity - 1;
end

% Constructs a B with possibly smaller size if we have detected higher nullity,
% otherwise B = Apert.

B = [1/sqrt(degq-d+1) * polytoep(pp, degq-d) ... % Sylvester matrix
    1/sqrt(degp-d+1) * polytoep(qq, degp-d)];

% This new Sylvester matrix should have nullity = 1. 
[~, ~, W] = svd(B);
x = W(:, end);

uu = x(1:degq-d+1);
vv = x(degq-d+2:end);
gg = polytoep(vv, d) \ pp;  % gg = deconv(pp, vv) but more stable

nearness = norm([conv(gg,vv)-p; conv(gg,-uu)-q])
d