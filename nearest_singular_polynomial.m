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
    if norm(y) ~= 0
        error('A nonzero value of y is not supported')
    end
    if ~isfield(store, 'r')
        n = size(A, 1);
        d = size(V, 2) - 1;
        M = zeros(k+d+1, (k+1)*n);
        W = zeros(size(V) + [0,2*k]);
        W(:,k+1:end-k) = V;
        % TODO: works only for k=2
        M(1:m+2,:) = [W(:,3:m+4).' W(:,2:m+3).' W(:,1:m+2).'];
        store.M = M;

        Av = A * M.';
        store.Av = Av;

        store.r = -Av - epsilon * y;
        s2 = svd(M);
        dd = 1./ (s2.^2 + epsilon);
        dd(~isfinite(dd)) = 0;
        store.d = dd;

        % normAv and condM are needed by penalty_method for stats
        store.normAv = norm(Av);
        store.condM = max(s2) / min(s2);
    end    
end

function [f, store] = cost(A, epsilon, y, V, store)
    store = populate_store(A, epsilon, y, V, store);
    M = store.M;
    f = real(trace((A.')'*((M'/(M*M'+epsilon*eye(k+d+1)))*(M*A.'))));

end

function [g, store] = egrad(A, epsilon, y, V, store)
    store = populate_store(A, epsilon, y, V, store);
    M = store.M;
    
    M_reg = M*M'+epsilon*eye(k+d+1);
    M_reg_inv = (M_reg^0) / M_reg;
    x = (M'/(M*M'+epsilon*eye(k+d+1)))*(M*A.');
    
    grad_M = 2*(M_reg_inv*M*A.'*(A.'-x)');
    % TODO: works only for k=2
    g = (grad_M(1:m,1:n) + grad_M(2:m+1,n+1:2*n) + grad_M(3:m+2,2*n+1:3*n)).';

end


function [H, store] = ehess(A, epsilon, y, V, dV, store)
    store = populate_store(A, epsilon, y, V, store);
        
    M = store.M;
    dW = zeros(size(V) + [0,2*k]);
    dW(:,k+1:end-k) = dV;
    % TODO: works only for k=2
    dM(1:m+2,:) = [dW(:,3:m+4).' dW(:,2:m+3).' dW(:,1:m+2).'];

    M_reg = M*M'+epsilon*eye(d+k+1);
    M_reg_inv = (M_reg^0) / M_reg;
    
    D_inv = - (M_reg_inv*(dM*M'+M*dM'))*M_reg_inv; 
    E = (M'/(M*M'+epsilon*eye(d+k+1)))*(M*A.');
       
    term1 = D_inv*M*A.'*(A.'-E)';
    term2 = M_reg_inv*dM*A.'*(A.'-E)';
    term3 = - M_reg_inv*M*A.'*(dM'*M_reg_inv*M*A.' + M'*D_inv*M*A.' + M'*M_reg_inv*dM*A.')';

    H_M = 2*(term1 + term2 + term3);
    % TODO: works only for k=2
    H = (H_M(1:m,1:n) + H_M(2:m+1,n+1:2*n) + H_M(3:m+2,2*n+1:3*n)).';

end

function E = minimizer(A, epsilon, y, V, store)
    store = populate_store(A, epsilon, y, V, store);
    M = store.M;
    
    E = -(M'/(M*M'+epsilon*eye(d+k+1)))*(M*A.');
    E = E.';

end

function [prod, store] = constraint(A, epsilon, y, V, store)
    store = populate_store(A, epsilon, y, V, store);
    M = store.M;
    E = -(M'/(M*M'+epsilon*eye(d+k+1)))*(M*A.');
    E = E.';

    prod = (A + E)*M.';
end

% function E = recover_exact(A, V, tol)
%     store = populate_store(A, epsilon, y, V, store);
%     M = store.M;
% 
%     E = (A*M.')*pinv(M.',tol);
% 
% end


end
