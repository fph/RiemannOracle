function problem = nearest_singular_polynomial(A, d, use_hessian)
% Nearest matrix polynomial with a right kernel of degree at most d
% For the moment it works only for degree(A) k=2.
% A = [A0 A1 A2] is given as input.
% The augmented Lagrangian method (y~=0) is not supported for now

k = 2; % hardcoded for now
n = size(A,1);

if not(exist('d', 'var')) || isempty(d)
    d = floor((k*(n-1))/2);
end

m = d+1;

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
        n = size(A, 1);
        d = size(V, 2) - 1;

        M = polytoep(reshape(V,[1,n,d+1]), k);
        store.M = M;

        % for this problem, the vector alpha such that T(A)v = M*alpha is
        % precisely A.', hence many transformations between matrices and their
        % basis representation reduce to transpositions.

        Av = M * A.';
        store.Av = Av;
        
        r = -Av - epsilon * y;
        z = (M*M'+epsilon*eye(k+d+1)) \ r;
        delta = M' * z;
       
        s2 = svd(M);
        % normAv and condM are needed by penalty_method for stats
        store.normAv = norm(Av,'f');
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
    M = store.M;
    delta = store.delta;
    
    grad_M = -2*(store.z*(A.'+delta)');
    % TODO: works only for k=2
    g = (grad_M(1:m,1:n) + grad_M(2:m+1,n+1:2*n) + grad_M(3:m+2,2*n+1:3*n)).';
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
    
    term1 = D_inv*(-r)*(A.'+delta)';
    term2 = M_reg_inv*dM*A.'*(A.'+delta)';
    term3 = z*(-dM'*z + M'*D_inv*(-r) + M'*M_reg_inv*dM*A.')';

    H_M = 2*(term1 + term2 + term3);
    % TODO: works only for k=2
    H = (H_M(1:m,1:n) + H_M(2:m+1,n+1:2*n) + H_M(3:m+2,2*n+1:3*n)).';
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
