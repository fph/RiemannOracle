function problem = nearest_defective_structured_dense(P, A)
% Create a Manopt problem structure for the nearest defective structured
% matrix.
% Uses a dense mxnxp array P as storage for the perturbation basis
% We assume (without checking) that this basis is orthogonal.

n = size(P, 2);
problem.M = stiefelcomplexfactory(n,2);

% populate the struct with generic functions that include the regularization as parameters
problem.gencost  = @(epsilon, y, v, store) cost(P, A, epsilon, y, v, store);
problem.genegrad = @(epsilon, y, v, store) egrad(P, A, epsilon, y, v, store);
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
    if isa(V, 'sym') % special code path for vpa(); useful for testing
        M1 = zeros(size(P,1), size(P,3), 'like', V);
        M2 = zeros(size(P,1), size(P,3), 'like', V);
        for i = 1:size(P,3)
            M1(:,i) = P(:,:,i)*V(:,2);
            M2(:,i) = transpose(P(:,:,i)) * conj(V(:,1));
        end
    else
        M1 = reshape(pagemtimes(P, V(:,2)), [size(P,1) size(P,3)]);
        M2 = reshape(pagemtimes(V(:,1)',P), [size(P,2) size(P,3)]);
    end
    M = [M1;M2];
end

% create Delta = \sum P(i)delta(i)
function Delta = make_Delta(P, delta)
    if isa(delta, 'sym')
        Delta = zeros(size(P,1),size(P,2), 'like', delta);
        for i = 1:size(P,3)
            Delta = Delta + delta(i)*P(:,:,i);
        end
    else
        Delta = tensorprod(P, delta, 3, 1);
    end
end

% fill values in the 'store', a caching structure used by Manopt
% with precomputed fields that we need in other functions as well
function store = populate_store(P, A, epsilon, y, V, store)
    if ~isfield(store, 'cf')
        if isscalar(y)  % in case we initialize with y = 0
            y = ones(2*size(P,1), 1) * y;
        end
        M = make_M(P, V);
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
            Av = A * V(:,2);
            store.normAv = norm(Av);
            r1 = [-Av;-transpose(A)*conj(V(:,1))] - y*epsilon;
            r2 = 0;
            r0 = r1;
        end
        store.s = s;
        d = 1 ./ (s.^2 + epsilon);
        store.d = d;
        [z0, delta0] = solve_system_svd(U1, WS, d, epsilon, r1, r2);
        vu = [V(:,2);conj(V(:,1))];
        store.vu = vu;
        [zp, deltap, a] = solve_system_svd(U1, WS, d, epsilon, vu, 0);
        lambda = -(vu'*z0) / a;
        %        r = lambda*vu + r0;
        
        %        z = lambda*zp + z0;
        %        delta = lambda*deltap + delta0;
        %        cf = real(r'*z);
        % this is more accurate because in some cases zp, z0 tend to get
        % much larger than z
        [z delta cf] = solve_system_svd(U1, WS, d, epsilon, r1+lambda*vu, r2);

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
    eg = zeros(n,2, 'like', V);
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
    prod = prod - store.vu*store.lambda;
end
