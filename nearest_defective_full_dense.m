function problem = nearest_defective_full_dense(P, A)
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
        assert(all(y==0)); % TODO: do general case
        Av = A * V(:,2);
%        r0 = [-Av;-transpose(A)*conj(V(:,1))] - y*epsilon;
%        r1 = [V(:,2); conj(V(:,1))];

%        n = size(V,1);
        Av = A*V(:,2);
        vAu = V(:,1)'*Av;
%        invMr0 = 1/(1+epsilon)*r0 + 1/(1+epsilon)/(2+epsilon)*vAu*[V(:,1);conj(V(:,2))];
%        invMr1 = 1/(1+epsilon)*r1;
        lambda = (V(:,2)'*A*V(:,2) + V(:,1)'*A*V(:,1)) / 2;
        %r = r0 + lambda*r1;
        %z = invMr0 + lambda * invMr1;

        v1 = Av - V(:,2)*lambda;
        u1 = V(:,1)'*A - lambda*V(:,1)';
        u1mod = u1 - (u1*V(:,2))*V(:,2)';
        cf = (norm(v1)^2 + norm(u1mod)^2 + epsilon/(2+epsilon)*abs(vAu)^2) / (1+epsilon);
        u1mod = (u1 - 1/(2+epsilon) * (u1*V(:,2))*V(:,2)') / (1+epsilon);
        v1mod = (v1 - 1/(2+epsilon) * V(:,1) * (V(:,1)'*v1)) / (1+epsilon);
        z = [-v1mod; transpose(-u1mod)];
        % z = [-v1mod; transpose(-u1mod)]
        %delta = kron(conj(V(:,2)), z(1:n)) + kron(z(n+1:end),V(:,1));
%        AplusDelta = A + make_Delta(P, delta);
        Delta = -V(:,1) * u1mod  - v1mod * V(:,2)';
        AplusDelta = A + Delta;

        store.normAv = norm(Av);
        store.lambda = lambda;
        store.z = z;
        store.Delta = Delta;
        store.AplusDelta = AplusDelta;
        store.cf = cf;
        store.condM = NaN;
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
    Delta = store.Delta;
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
        prod = store.M * (A + store.Delta);
    else
        prod = [AplusDelta * V(:,2); transpose(AplusDelta)*conj(V(:,1))];
    end
    prod = prod - [V(:,2); conj(V(:,1))]*store.lambda;
end
