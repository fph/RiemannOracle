function problem = nearest_singular_polynomial(A, d, use_hessian)
% Nearest matrix polynomial with a right kernel of degree at most d.
% 
% The input A can either be in the form of a m x n x (k+1) array, 
% or as A = [A0 A1 A2 ... Ak] (only for square polynomials, m=n)
% 
% k = degree(A), and the leading term is in Ak or A(:,:,k+1).

if ismatrix(A)
    [m, nkplus1] = size(A);
    n = m;
    k = nkplus1 / n - 1;
else % preliminary support for rectangular matrix polynomials, given as a mxnx(k+1) array
    [m, n, kplus1] = size(A);
    k = kplus1 - 1;
    A = reshape(A, m, n*kplus1);
end

if not(exist('d', 'var')) || isempty(d)
    % this is a sensible default, but for a square polynomial, it is better to 
    % solve two problems for the left and right kernel and degree floor(k(n-1)/2).
    d = k*(n-1);
end

if isreal(A)
    problem.M = spherefactory(n, d+1);
else
    problem.M = spherecomplexfactory(n, d+1);
end

% populate the struct with generic functions that include the regularization as parameters
problem.gencost  = @(epsilon, y, v, store) cost(A, epsilon, y, v, store);
problem.genegrad = @(epsilon, y, v, store) egrad(A, epsilon, y, v, store);
if use_hessian
    problem.genehess = @(epsilon, y, v, w, store) ehess(A, epsilon, y, v, w, store);
end
problem.genminimizer = @(epsilon, y, v, store) minimizer(A, epsilon, y, v, store);

% compute the value of the constraint (A+E)*M.'.
% Note that this constraint is not zero when epsilon is nonzero
problem.genconstraint = @(epsilon, y, v, store) constraint(A, epsilon, y, v, store);

% populate functions from the Manopt interface with zero regularization
problem = apply_regularization(problem, 0, 0);

% additional function outside of the main interface to recover
% a more exact solution (not yet supported)
% problem.recover_exact = @(v, tol) recover_exact(A, v, tol);

% fill values in the 'store', a caching structure used by Manopt
% with precomputed fields that we need in other functions as well
function store = populate_store(A, epsilon, y, V, store)
    if ~isfield(store, 'cf')
        if isscalar(y)  % in case we initialize with y = 0
            y = ones(k+d+1, m) * y;
        end

        % This M is actually the matrix such the real M is kron(M,I)
        M = polytoep(reshape(V,[1,n,d+1]), k);
        [U1, S, W] = svd(M, 'econ');
        
        store.M = M;
        store.U1 = U1;
        s = diag(S);
        WS = W .* s';
        store.WS = WS;

        % we work implicitly with T(A) = polytoep(A, d)

        % for this problem, the vector alpha such that T(A)v = M*alpha 
        % and T(A) = sum_i P(:,:,i)*alpha_i  is
        % precisely A.', up to reshaping, hence many transformations 
        % between matrices and their basis representation reduce to transpositions + reshaping.
        % for instance, Delta (as a polynomial) is delta.',
        % and T(A)*v = vec(transpose(M * A.')).

        r1 = -epsilon * y;
        r2 = -WS' * (A.'); % = -U1' * (T(A)*v) = -U1' * M * (A.')
        % alternative formulas:
        % TAv = M * (A.');
        % r = -TAv - epsilon * y;
        % store.r = r;

        dg = 1 ./ (s.^2 + epsilon);
        store.d = dg;
        [z, delta, cf] = solve_system_svd(U1, WS, dg, epsilon, r1, r2);
        % alternative formulas:
        % z = (M*M'+epsilon*eye(k+d+1)) \ r;
        % delta = M' * z;
       
        % normAv and condM are needed by penalty_method for stats
        store.normAv = norm(r2,'f');
        store.condM = max(s) / min(s);
        store.z = z;
        store.delta = delta;
        store.AplusDelta = A + delta.';
        store.AplusDeltaTranspose = (store.AplusDelta).'; % we store both to save a transposition later
        store.cf = cf;
    end    
end

function [f, store] = cost(A, epsilon, y, V, store)
    store = populate_store(A, epsilon, y, V, store);
    f = store.cf;
    % alternative formulas:
    % f = real(sum(sum(conj(store.r) .* store.z)));  % = real(trace(r'*z))
end

function [g, store] = egrad(A, epsilon, y, V, store)
    store = populate_store(A, epsilon, y, V, store);

    g = polytoep_adjoint_vec(reshape(store.AplusDelta,[m,n,k+1]), d, -2*transpose(store.z));
    g = reshape(g, [n,d+1]);
end


function [H, store] = ehess(A, epsilon, y, V, dV, store)
    store = populate_store(A, epsilon, y, V, store);

    M = store.M;
    z = store.z;
    WS = store.WS;
    dM = polytoep(reshape(dV,[1,n,d+1]), k);
    r1 = dM * store.AplusDeltaTranspose;
    r2 = WS' * (dM'*z);
    dz = -solve_system_svd(store.U1, WS, store.d, epsilon, r1, r2);

%    alternative formulas:
%    M_reg = M*M'+epsilon*eye(d+k+1);
%    M_reg_inv = (M_reg^0) / M_reg;   
%    D_inv = - (M_reg_inv*(dM*M'+M*dM'))*M_reg_inv; 
%    dz = D_inv*store.r - M_reg_inv*dM*A.';

    ddelta = dM'*z + M'*dz;
    t1 = polytoep_adjoint_vec(reshape(store.AplusDelta,[m,n,k+1]), d, transpose(dz));
    t2 = polytoep_adjoint_vec(reshape(ddelta.',[m,n,k+1]), d, transpose(z));
    H = -2*(t1 + t2);
    H = reshape(H, [n,d+1]);
end

function [Delta, AplusDelta, store] = minimizer(A, epsilon, y, V, store)
    store = populate_store(A, epsilon, y, V, store);
    Delta = store.delta.';
    AplusDelta = store.AplusDelta;
end

function [prod, store] = constraint(A, epsilon, y, V, store)
    store = populate_store(A, epsilon, y, V, store);
    prod = store.M * store.AplusDeltaTranspose;
end

% function E = recover_exact(A, V, tol)
%     store = populate_store(A, epsilon, y, V, store);
%     M = store.M;
% 
%     E = (A*M.')*pinv(M.',tol);
% 
% end


end
