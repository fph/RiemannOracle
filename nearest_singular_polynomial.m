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
    if ~isfield(store, 'r')
        d = size(V, 2) - 1;

        M = polytoep(reshape(V,[1,n,d+1]), k);
        store.M = M;
        % we work implicitly with T(A) = polytoep(A, d)

        % for this problem, the vector alpha such that T(A)v = M*alpha 
        % and T(A) = sum_i P(:,:,i)*alpha_i  is
        % precisely A.', up to reshaping, hence many transformations 
        % between matrices and their basis representation reduce to transpositions + reshaping.
        % for instance, Delta (as a polynomial) is delta.',
        % and T(A)*v = vec(transpose(M * A.')).

        TAv = M * A.';
        
        r = -TAv - epsilon * y;
        z = (M*M'+epsilon*eye(k+d+1)) \ r;
        delta = M' * z;
       
        s2 = svd(M);
        % normAv and condM are needed by penalty_method for stats
        store.normAv = norm(TAv,'f');
        store.condM = max(s2) / min(s2);
        store.r = r;
        store.z = z;
        store.delta = delta;
    end    
end

function [f, store] = cost(A, epsilon, y, V, store)
    store = populate_store(A, epsilon, y, V, store);
    f = real(trace(store.r'*store.z));
end

function [g, store] = egrad(A, epsilon, y, V, store)
    store = populate_store(A, epsilon, y, V, store);
    delta = store.delta;
    
    g = polytoep_adjoint_vec(reshape(A+delta.',[m,n,k+1]), d, -2*transpose(store.z));
    g = reshape(g, [n,d+1]);
end


function [H, store] = ehess(A, epsilon, y, V, dV, store)
    store = populate_store(A, epsilon, y, V, store);

    M = store.M;
    r = store.r;
    z = store.z;
    dM = polytoep(reshape(dV,[1,n,d+1]), k);

    M_reg = M*M'+epsilon*eye(d+k+1);
    M_reg_inv = (M_reg^0) / M_reg;
    
    D_inv = - (M_reg_inv*(dM*M'+M*dM'))*M_reg_inv; 
    delta = store.delta;
    dz = D_inv*r - M_reg_inv*dM*A.';
    ddelta = dM'*z + M'*dz;
    
    t1 = polytoep_adjoint_vec(reshape(A+delta.',[m,n,k+1]), d, transpose(dz));
    t2 = polytoep_adjoint_vec(reshape(ddelta.',[m,n,k+1]), d, transpose(z));
    H = -2*(t1 + t2);
    H = reshape(H, [n,d+1]);
end

function Delta = minimizer(A, epsilon, y, V, store)
    store = populate_store(A, epsilon, y, V, store);
    Delta = store.delta.';
end

function [prod, store] = constraint(A, epsilon, y, V, store)
    store = populate_store(A, epsilon, y, V, store);
    M = store.M;
    Delta = store.delta.';
    prod = M * (A + Delta).';
end

% function E = recover_exact(A, V, tol)
%     store = populate_store(A, epsilon, y, V, store);
%     M = store.M;
% 
%     E = (A*M.')*pinv(M.',tol);
% 
% end


end
