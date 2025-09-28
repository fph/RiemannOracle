function problem = nearest_defective_structured_dense_new(P, A, use_hessian)
% Create a Manopt problem structure for the nearest defective structured
% matrix.
% Uses a dense mxnxp array P as storage for the perturbation basis
% We assume (without checking) that this basis is orthogonal.
%
% This variant uses some more optimizations specific to this problem

if not(exist('use_hessian', 'var'))
    use_hessian = false;
end
if use_hessian
    error('The exact Hessian for this problem is not known (to us).')
end

n = size(P, 2);
problem.M = stiefelcomplexfactory(n,2);

% populate the struct with generic functions that include the regularization as parameters
problem.gencost  = @(epsilon, y, v, store) cost(P, A, epsilon, y, v, store);
problem.genegrad = @(epsilon, y, v, store) egrad(P, A, epsilon, y, v, store);
if use_hessian
    problem.genehess = @(epsilon, y, v, w, store) ehess(P, A, epsilon, y, v, w, store);
end
problem.genminimizer = @(epsilon, y, v, store) minimizer(P, A, epsilon, y, v, store);

% compute the value of the constraint (A+E)v.
% We need this for the extended Lagrangian update
% Note that this constraint is not zero when epsilon is nonzero
problem.genconstraint = @(epsilon, y, v, store) constraint(P, A, epsilon, y, v, store);

% populate functions from the Manopt interface with zero regularization
problem = apply_regularization(problem, 0, 0);

end

% create M(v)
function M = make_M(P, V)
    M1 = reshape(pagemtimes(P, V(:,2)), [size(P,1) size(P,3)]);
    M2 = reshape(pagemtimes(V(:,1)',P), [size(P,2) size(P,3)]);
    M = [M1;M2];
end

% create Delta = \sum P(i)delta(i)
function Delta = make_Delta(P, delta)
    Delta = tensorprod(P, delta, 3, 1);
end

% fill values in the 'store', a caching structure used by Manopt
% with precomputed fields that we need in other functions as well
function store = populate_store(P, A, epsilon, y, V, store)
    if ~isfield(store, 'cf')
        if isscalar(y)  % in case we initialize with y = 0
            y = ones(2*size(P,1), 1) * y;
        end
        M = make_M(P, V);
        if epsilon==0
            % trick: we "regularize" M in case epsilon=0,
            % to change its structural zero singular value into an 1. This
            % modification
            % will not change the results of the following computations, since 
            % r is orthogonal to [-u; conj(v)]
            Mreg = M + [-V(:,1); conj(V(:,2))] * kron(transpose(V(:,1)), V(:,2)');
            [U1, S, W] = svd(Mreg, 'econ');
        else
            [U1, S, W] = svd(M, 'econ');
        end

        store.M = M;
        store.U1 = U1;
        s = diag(S);
        store.condM = max(s) / min(s);
        WS = W .* s';
        store.WS = WS;

        % shortcut: U1=U is square here, since M is always short-fat
        assert(size(U1,1)==size(U1,2));

        if isvector(A)
            SWTalpha = store.WS' * A;
            store.normAv = norm(SWTalpha);
            Utr0 = -SWTalpha - U1'*y*epsilon;
            Utr1 = U1'*[V(:,2); conj(V(:,1))];
        else
            Av = A * V(:,2);
            Utr0 = U1'*([-Av;-transpose(A)*conj(V(:,1))] - y*epsilon);
            store.normAv = norm(Av);
            Utr1 = U1'*[V(:,2); conj(V(:,1))];
        end
        store.s = s;
        d = 1 ./ (s.^2 + epsilon);
        store.d = d;
        % shortcut: lambda = lambda0, for this problem
        lambda = -sum(conj(Utr1) .* Utr0 .* d) / sum(conj(Utr1) .* Utr1 .* d);
        Utr = Utr1 * lambda + Utr0;
        cf = sum(conj(Utr) .* Utr .* d);
        z = U1 * (d .* Utr);
        delta = WS * (d .* Utr);

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

function [cf, store] = cost(P, A, epsilon, y, V, store)
    store = populate_store(P, A, epsilon, y, V, store);
    cf = store.cf;
end

function [eg, store] = egrad(P, A, epsilon, y, V, store)
    store = populate_store(P, A, epsilon, y, V, store);
    n = size(P,1);
    eg = zeros(n,2);
    lambda = store.lambda;
    AplusDelta = store.AplusDelta;
    z = store.z;
    eg(:,2) = (conj(lambda)*z(1:n) - AplusDelta'*z(1:n)) * 2;
    eg(:,1) = (lambda*conj(z(n+1:end)) - AplusDelta*conj(z(n+1:end))) * 2;
end

function [Delta, AplusDelta, store] = minimizer(P, A, epsilon, y, V, store)
    store = populate_store(P, A, epsilon, y, V, store);
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

function [prod, store] = constraint(P, A, epsilon, y, V, store)
    [Delta, AplusDelta, store] = minimizer(P, A, epsilon, y, V, store);
    if isvector(A)
        prod = store.M * (A + store.delta);
    else
        prod = [AplusDelta * V(:,2); transpose(AplusDelta)*conj(V(:,1))];
    end
    prod = prod - [V(:,2); conj(V(:,1))]*store.lambda;
end
