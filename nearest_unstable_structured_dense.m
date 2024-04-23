function problem = nearest_unstable_structured_dense(P, target, A)
% Create a Manopt problem structure for the nearest unstable structured
% matrix.
% Uses a dense mxnxp array P as storage for the perturbation basis
% We assume (without checking) that this basis is orthogonal.

n = size(P, 2);
problem.M = spherecomplexfactory(n);

if strcmp(target, 'Schur')
    target = @(x) outside_disc(x, 1);
elseif strcmp(target, 'Hurwitz')
    target = @(x) right_of(x, 0);
end

% populate the struct with generic functions that include the regularization as parameters
problem.gencost  = @(epsilon, y, v, store) cost(P, target, A, epsilon, y, v, store);
problem.genegrad = @(epsilon, y, v, store) egrad(P, target, A, epsilon, y, v, store);
problem.genminimizer = @(epsilon, y, v, store) minimizer(P, target, A, epsilon, y, v, store);

% compute the value of the constraint (A+E)v + epsilon*y.
% We need this for the extended Lagrangian update
% Note that this constraint is not zero when epsilon is nonzero
problem.genconstraint = @(epsilon, y, v, store) constraint(P, target, A, epsilon, y, v, store);

% populate functions from the Manopt interface with zero regularization
problem = apply_regularization(problem, 0, 0);

end

% create M(v)
function M = make_M(P, v)
    M = reshape(pagemtimes(P, v), [size(P,1) size(P,3)]);
end

% create Delta = \sum P(i)delta(i)
function Delta = make_Delta(P, delta)
    Delta = tensorprod(P, delta, 3, 1);
end

% fill values in the 'store', a caching structure used by Manopt
% with precomputed fields that we need in other functions as well
function store = populate_store(P, target, A, epsilon, y, v, store)
    if ~isfield(store, 'cf')
        if isscalar(y)  % in case we initialize with y = 0
            y = ones(size(P,1), 1) * y;
        end
        M = make_M(P, v);
        [U1, S, W] = svd(M, 'econ');
        
        store.M = M;
        store.U1 = U1;
        s = diag(S);
        store.condM = max(s) / min(s);
        WS = W .* s';
        store.WS = WS;
        if isvector(A)
            SWTalpha = store.WS' * A;
            store.normAv = norm(SWTalpha);
            r1 = -y*epsilon;
            r2 = -SWTalpha;
            r0 = r1 + U1*r2;
        else
            Av = A * v;
            store.normAv = norm(Av);
            r1 = -Av - y*epsilon;
            r2 = 0;
            r0 = r1;
        end
        store.s = s;
        d = 1 ./ (s.^2 + epsilon);
        store.d = d;
        [z0, delta0] = solve_system_svd(U1, WS, d, epsilon, r1, r2);
        [zp, deltap, a] = solve_system_svd(U1, WS, d, epsilon, v, 0);
        lambda0 = -(v'*z0) / a;
        lambda = target(lambda0);
        r = lambda*v + r0;
        z = lambda*zp + z0;
        delta = lambda*deltap + delta0;
        cf = real(r'*z);

        if isvector(A)
            AplusDelta = make_Delta(P, A+delta);
        else
            AplusDelta = A + make_Delta(P, delta);
        end
        store.lambda = lambda;
        store.z = z;
        store.delta = delta;
        store.AplusDelta = AplusDelta;
        store.cf = cf;
    end
end

function [cf, store] = cost(P, target, A, epsilon, y, v, store)
    store = populate_store(P, target, A, epsilon, y, v, store);
    cf = store.cf;
end

function [eg, store] = egrad(P, target, A, epsilon, y, v, store)
    store = populate_store(P, target, A, epsilon, y, v, store);
    z = store.z;
    eg = (conj(store.lambda)*z - store.AplusDelta'*z) * 2;
end

function [Delta, AplusDelta, store] = minimizer(P, target, A, epsilon, y, v, store)
    store = populate_store(P, target, A, epsilon, y, v, store);
    Delta = make_Delta(P, store.delta);
    AplusDelta = store.AplusDelta;
end

% function [ehw, store] = ehess(P, A, epsilon, y, v, w, store)
%     store = populate_store(P, A, epsilon, y, v, store);
%     AplusDelta = store.AplusDelta;
%     z = store.z;
%     M = store.M;
%     WS = store.WS;
%     dM = make_M(P, w);
%     dz = -solve_system_svd(store.U1, WS, store.d, epsilon, AplusDelta*w, WS'*(dM'*z));
%     ddelta = dM' * z + M' * dz;
%     dDelta = make_Delta(P, ddelta);
%     ehw = (dDelta' * z + AplusDelta' * dz) * (-2);
% end

function [prod, store] = constraint(P, target, A, epsilon, y, v, store)
    [Delta, AplusDelta, store] = minimizer(P, target, A, epsilon, y, v, store);
    if isvector(A)
        prod = store.M * (A + store.delta);
    else
        prod = AplusDelta * v;
    end
    prod = prod - v*store.lambda;
end
