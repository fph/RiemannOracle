function [v delta] = stln_gaussnewton(problem, options)

epsilon = 1e-4;
A = problem.A;
[m, n] = size(A);

v = randn(n, 1);
[Delta, AplusDelta, store] = problem.minimizer(v, []);
M = store.M;
p = size(M, 2);
delta = randn(p, 1);

for k = 1:50
    
    Delta = problem.make_Delta(delta);
    M = problem.make_M(v);
    resv = norm((A+Delta)*v)
    normdelta = norm(delta)

    W = [M A+Delta; epsilon*eye(p) zeros(p, n)];
    rhs = [-(A+Delta)*v; -epsilon*delta];
    sol = W \ rhs;  % least-squares problem
    normsol = norm(sol)
    res = norm(W*sol-rhs)
    delta = delta + sol(1:p);
    v = v + sol(p+1:end);
    normv = norm(v)
    v = v / norm(v);
end