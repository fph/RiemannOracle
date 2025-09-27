function problem = nearest_defective_full_dense(A, use_hessian)
% Create a Manopt problem structure for the nearest defective structured
% matrix.
% Uses a dense mxnxp array P as storage for the perturbation basis
% We assume (without checking) that this basis is orthogonal.

if not(exist('use_hessian', 'var'))
    use_hessian = false;
end
if use_hessian
    error('The exact Hessian for this problem is not known (to us).')
end

n = size(A, 2);
problem.M = stiefelcomplexfactory(n,2);

% populate the struct with generic functions that include the regularization as parameters
problem.gencost  = @(epsilon, y, v, store) cost(A, epsilon, y, v, store);
problem.genegrad = @(epsilon, y, v, store) egrad(A, epsilon, y, v, store);
if use_hessian
    problem.genehess = @(epsilon, y, v, w, store) ehess(P, A, epsilon, y, v, w, store);
end
problem.genminimizer = @(epsilon, y, v, store) minimizer(A, epsilon, y, v, store);

% compute the value of the constraint (A+E)v.
% We need this for the extended Lagrangian update
% Note that this constraint is not zero when epsilon is nonzero
problem.genconstraint = @(epsilon, y, v, store) constraint(A, epsilon, y, v, store);

% populate functions from the Manopt interface with zero regularization
problem = apply_regularization(problem, 0, 0);

end

% fill values in the 'store', a caching structure used by Manopt
% with precomputed fields that we need in other functions as well
function store = populate_store(A, epsilon, y, V, store)
    if ~isfield(store, 'cf')
        if isscalar(y)  % in case we initialize with y = 0
            y = ones(2*size(A,1), 1) * y;
        end
        
        n = size(V,1);
        v = V(:,2);
        u = V(:,1);
        Av = A*v;
        Astaru = A'*u;
        r1 = [v; conj(u)];
        r0 = -[Av; conj(Astaru)] - y*epsilon;
        beta = r1'*r0;
        yv = y(1:n);
        conjyu = conj(y(n+1:end));
        gamma1 = conj(v'*(Astaru + conjyu*epsilon));
        gamma2 = u'*(Av+yv*epsilon);
        a = 2/(1+epsilon); b = beta/(1+epsilon);
        lambda = -b/a;
        if epsilon == 0
            secondterm = 0;
        else
            secondterm = (gamma1-gamma2) / epsilon / (2+epsilon);
        end
        eta1 = gamma1/(2+epsilon) + secondterm;
        eta2 = gamma2/(2+epsilon) - secondterm;
        %eta1 = ((1+epsilon)*gamma1 - gamma2) / (epsilon*(2+epsilon));
        %eta2 = ((1+epsilon)*gamma2 - gamma1) / (epsilon*(2+epsilon));
        % TODO: recheck sign in front of eta1, et2 with theory!
        zv = 1/(1+epsilon) * (lambda*v-Av - yv*epsilon + u*eta1); 
        zu = 1/(1+epsilon) * (conj(lambda)*u-Astaru - conjyu*epsilon + v*conj(eta2));
        z = [zv; conj(zu)];
        r = r1 * lambda + r0;
        assert(imag(r'*z) < 1e-6);
        cf = real(r'*z);

        Delta = u * zu'  + zv * v';
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

function [cf, store] = cost(A, epsilon, y, V, store)
    store = populate_store(A, epsilon, y, V, store);
    cf = store.cf;
end

function [eg, store] = egrad(A, epsilon, y, V, store)
    store = populate_store(A, epsilon, y, V, store);
    n = size(A,1);
    eg = zeros(n,2);
    lambda = store.lambda;
    AplusDelta = store.AplusDelta;
    z = store.z;
    eg(:,2) = (conj(lambda)*z(1:n) - AplusDelta'*z(1:n)) * 2;
    eg(:,1) = (lambda*conj(z(n+1:end)) - AplusDelta*conj(z(n+1:end))) * 2;
end

function [Delta, AplusDelta, store] = minimizer(A, epsilon, y, V, store)
    store = populate_store(A, epsilon, y, V, store);
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

function [prod, store] = constraint(A, epsilon, y, V, store)
    [Delta, AplusDelta, store] = minimizer(A, epsilon, y, V, store);
    if isvector(A)
        prod = store.M * (A + store.Delta);
    else
        prod = [AplusDelta * V(:,2); transpose(AplusDelta)*conj(V(:,1))];
    end
    prod = prod - [V(:,2); conj(V(:,1))]*store.lambda;
end
