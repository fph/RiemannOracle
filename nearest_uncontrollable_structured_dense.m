function problem = nearest_uncontrollable_structured_dense(P, A, use_hessian)
% Create a Manopt problem structure for the nearest uncontrollable structured
% matrix.
% Uses a dense m x nk x p array P as storage for the perturbation basis
% We assume (without checking) that this basis is orthogonal.
% Input A should be a tensor of size m x n x k.

[~,n,~] = size(A);
problem.M = productmanifold(struct('X', euclideancomplexfactory(1), 'Y', spherecomplexfactory(n)));

% populate the struct with generic functions that include the regularization as parameters
problem.gencost  = @(epsilon, y, v, store) cost(P, A, epsilon, y, v, store);
problem.genegrad = @(epsilon, y, v, store) egrad(P, A, epsilon, y, v, store);
if use_hessian
    problem.genehess = @(epsilon, y, v, w, store) ehess(P, A, epsilon, y, v, w, store);
end
problem.genminimizer = @(epsilon, y, v, store) minimizer(P, A, epsilon, y, v, store);

% compute the value of the constraint (A+E)v + epsilon*y.
% We need this for the extended Lagrangian update
% Note that this constraint is not zero when epsilon is nonzero
problem.genconstraint = @(epsilon, y, v, store) constraint(P, A, epsilon, y, v, store);

% populate functions from the Manopt interface with zero regularization
problem = apply_regularization(problem, 0, 0);

% additional function outside of the main interface to recover
% a more exact solution in the case of rank drops, enforcing exact
% zeros in U'*r
problem.recover_exact = @(v, tol) recover_exact(P, A, v, tol);

end

% create M(v)
function M = make_M(P, v, k)
    w = kron(v.X.^(k-1:-1:0).',v.Y);
    M = reshape(pagemtimes(P, w), [size(P,1) size(P,3)]);
end

function M = make_M_prime(P, v, k)
    w = kron([((k-1:-1:1).*v.X.^(k-2:-1:0)) 0].',v.Y);
    M = reshape(pagemtimes(P, w), [size(P,1) size(P,3)]);
end

% create Delta = \sum P(i)delta(i)
function Delta = make_Delta(P, delta)
    Delta = tensorprod(P, delta, 3, 1);
end

function Av = evalAv(A, v, lambda, k)
    w = kron(lambda.^(k-1:-1:0).',v);
    Av = A * w;
end

function Av = evalAprimev(A, v, lambda, k)
    w = kron([((k-1:-1:1).*lambda.^(k-2:-1:0)) 0].',v);
    Av = A * w;
end

% fill values in the 'store', a caching structure used by Manopt
% with precomputed fields that we need in other functions as well
function store = populate_store(P, A, epsilon, y, v, store)
    if ~isfield(store, 'A')
        [m,n,k] = size(A);
        store.k = k;
        store.n = n;
        store.m = m;
        store.A = reshape(A,m,n*k);
    end

    if ~isfield(store, 'cf')
        if isscalar(y)  % in case we initialize with y = 0
            y = ones(size(P,1), 1) * y;
        end
        M = make_M(P, v, store.k);
        [U1, S, W] = svd(M, 'econ');
        
        store.M = M;
        store.U1 = U1;
        s = diag(S);
        store.condM = max(s) / min(s);
        WS = W .* s';
        store.WS = WS;
        if isvector(store.A)
            SWTalpha = store.WS' * store.A;
            store.normAv = norm(SWTalpha);
            r1 = -y*epsilon;
            r2 = -SWTalpha;
        else
            Av = evalAv(store.A, v.Y, v.X, store.k);
            store.normAv = norm(Av);
            r1 = -Av - y*epsilon;
            r2 = 0;
        end
        store.s = s;
        d = 1 ./ (s.^2 + epsilon);
        store.d = d;
        [z, delta, cf] = solve_system_svd(U1, WS, d, epsilon, r1, r2);
        if isvector(store.A)
            AplusDelta = make_Delta(P, store.A+delta);
        else
            wI = kron(v.X.^(store.k-1:-1:0).',eye(size(P,2)/k));
            AplusDelta = (store.A + make_Delta(P, delta))*wI;
        end
        store.z = z;
        store.delta = delta;
        store.AplusDelta = AplusDelta;
        store.cf = cf;
    end
end

function [cf, store] = cost(P, A, epsilon, y, v, store)
    store = populate_store(P, A, epsilon, y, v, store);
    cf = store.cf;
end

function [eg, store] = egrad(P, A, epsilon, y, v, store)
    store = populate_store(P, A, epsilon, y, v, store);
    Delta = make_Delta(P, store.delta);
    w = kron([((store.k-1:-1:1).*v.X.^(store.k-2:-1:0)) 0].',v.Y);
    eg.X = ((store.A + Delta)*w)' * store.z * (-2);
    eg.Y = store.AplusDelta' * store.z * (-2);
end

function [Delta, AplusDelta, store] = minimizer(P, A, epsilon, y, v, store)
    store = populate_store(P, A, epsilon, y, v, store);
    Delta = make_Delta(P, store.delta);
    AplusDelta = store.AplusDelta;
end

function [ehw, store] = ehess(P, A, epsilon, y, v, w, store)
    store = populate_store(P, A, epsilon, y, v, store);
    AplusDelta = store.AplusDelta;
    z = store.z;
    M = store.M;
    WS = store.WS;
    vxwy = struct();
    vxwy.X = v.X;
    vxwy.Y = w.Y;
    dM = make_M(P, vxwy, store.k);
    dz = -solve_system_svd(store.U1, WS, store.d, epsilon, AplusDelta*w.Y, WS'*(dM'*z));
    ddelta = dM' * z + M' * dz;
    dDelta = make_Delta(P, ddelta);
    
    M_prime = make_M_prime(P, v, store.k)*w.X;
    Delta = make_Delta(P, store.delta);
    dv = kron([((store.k-1:-1:1).*v.X.^(store.k-2:-1:0)) 0].',v.Y);
    dw = kron([((store.k-1:-1:1).*v.X.^(store.k-2:-1:0)) 0].',w.Y);
    z_prime = -solve_system_svd(store.U1, WS, store.d, epsilon, (store.A + Delta)*dv*w.X, WS'*(M_prime'*z));
    
    delta_prime = M_prime'*z + M'*z_prime;
    Delta_prime1 = make_Delta(P, delta_prime);
    Delta_prime2 = evalAprimev(Delta, eye(size(P,2)/store.k), v.X, store.k)*w.X;
    Delta_prime = Delta_prime2 + evalAv(Delta_prime1, eye(size(P,2)/store.k), v.X, store.k);

    dDeltaHz = evalAv(dDelta, eye(size(P,2)/store.k), v.X, store.k)'*z;
    ddv = kron([((store.k-2:-1:1).*(store.k-1:-1:2).*v.X.^(store.k-3:-1:0)) 0 0].',v.Y);
    AplusDelta_prime = evalAprimev(store.A, eye(size(P,2)/store.k), v.X, store.k)*w.X + Delta_prime;
    
    ehw.Y = (AplusDelta_prime'*store.z + AplusDelta'*z_prime + dDeltaHz + AplusDelta' * dz) * (-2);
    ehw.X = (((store.A + Delta)*ddv*w.X + evalAprimev(Delta_prime1, v.Y, v.X, store.k))'*z + ((store.A + Delta)*dv)'*z_prime ... 
        + ((store.A + Delta)*dw)'*z + (dDelta*dv)'*z + ((store.A + Delta)*dv)'*dz) * (-2); 

end

function [prod, store] = constraint(P, A, epsilon, y, v, store)
    [Delta, AplusDelta, store] = minimizer(P, A, epsilon, y, v, store);
    if isvector(A)
        prod = store.M * (A + store.delta);
    else
        prod = AplusDelta * v.Y;
    end
end

function v_reg = recover_exact(P, A, v, tol)
    TODO %: trickier since U is not constant anymore
    store = struct();
    store = populate_store(P, A, 0, 0, v, store);
    r = store.r;
    d = store.d;
    r_reg = r;
    r_reg(abs(d) > tol) = 0;
    v_reg = - A \ r_reg;
end
