function problem = nearest_singular_polynomial_compatible(A, use_hessian)
% Augmented Lagrangian not supported

n = size(A,1);
m = floor((2*(n-1))/2)+1;

if isreal(A)
    problem.M = spherefactory(n,m);
else
    problem.M = spherecomplexfactory(n,m);
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
        M = zeros(m+2,3*n);
        W = zeros(size(V) + [0,4]);
        W(:,3:end-2) = V;
        
        M(1:m+2,:) = [W(:,3:m+4).' W(:,2:m+3).' W(:,1:m+2).'];

        Av = A * M.';
        store.Av = Av;

        store.r = -Av - epsilon * y;
        s2 = svd(M);
        d = 1./ (s2.^2 + epsilon);
        d(~isfinite(d)) = 0;
        store.d = d;

        % normAv and condM are needed by penalty_method for stats
        store.normAv = norm(Av);
        store.condM = max(s2) / min(s2);
    end    
end

function [f, store] = cost(A, epsilon, y, V, store)
    store = populate_store(A, epsilon, y, V, store);
    
    M = zeros(m+2,3*n);
    W = zeros(size(V) + [0,4]);
    W(:,3:end-2) = V;
    
    M(1:m+2,:) = [W(:,3:m+4).' W(:,2:m+3).' W(:,1:m+2).'];
    
    f = real(trace((A.')'*((M'/(M*M'+epsilon*eye(m+2)))*(M*A.'))));

end

function [g, store] = egrad(A, epsilon, y, V, store)
    store = populate_store(A, epsilon, y, V, store);

    n = size(A,1);
    m = floor((2*(n-1))/2)+1;

    M = zeros(m+2,3*n);
    W = zeros(size(V) + [0,4]);
    W(:,3:end-2) = V;

    M(1:m+2,:) = [W(:,3:m+4).' W(:,2:m+3).' W(:,1:m+2).'];
    
    M_reg = M*M'+epsilon*eye(m+2);
    M_reg_inv = (M_reg^0) / M_reg;
    x = (M'/(M*M'+epsilon*eye(m+2)))*(M*A.');
    
    grad_M = 2*(M_reg_inv*M*A.'*(A.'-x)');
    g = (grad_M(1:m,1:n) + grad_M(2:m+1,n+1:2*n) + grad_M(3:m+2,2*n+1:3*n)).';

end


function [H, store] = ehess(A, epsilon, y, V, dV, store)
    store = populate_store(A, epsilon, y, V, store);
    
    n = size(A,1);
    m = floor((2*(n-1))/2)+1;
    
    M = zeros(m+2,3*n);
    W = zeros(size(V) + [0,4]);
    W(:,3:end-2) = V;
    dW = zeros(size(V) + [0,4]);
    dW(:,3:end-2) = dV;
    
    M(1:m+2,:) = [W(:,3:m+4).' W(:,2:m+3).' W(:,1:m+2).'];
    d(1:m+2,:) = [dW(:,3:m+4).' dW(:,2:m+3).' dW(:,1:m+2).'];

    M_reg = M*M'+epsilon*eye(m+2);
    M_reg_inv = (M_reg^0) / M_reg;
    
    D_inv = - (M_reg_inv*(d*M'+M*d'))*M_reg_inv; 
    
    E = (M'/(M*M'+epsilon*eye(m+2)))*(M*A.');
       
    term1 = D_inv*M*A.'*(A.'-E)';
    term2 = M_reg_inv*d*A.'*(A.'-E)';
    term3 = - M_reg_inv*M*A.'*(d'*M_reg_inv*M*A.' + M'*D_inv*M*A.' + M'*M_reg_inv*d*A.')';

    H_M = 2*(term1 + term2 + term3);
    
    H = (H_M(1:m,1:n) + H_M(2:m+1,n+1:2*n) + H_M(3:m+2,2*n+1:3*n)).';

end

function E = minimizer(A, epsilon, y, V, store)
    n = size(A,1);
    m = floor((2*(n-1))/2)+1;
    
    M = zeros(m+2,3*n);
    W = zeros(size(V) + [0,4]);
    W(:,3:end-2) = V;

    M(1:m+2,:) = [W(:,3:m+4).' W(:,2:m+3).' W(:,1:m+2).'];
    
    E = -(M'/(M*M'+epsilon*eye(m+2)))*(M*A.');
    E = E.';

end

function [prod, store] = constraint(A, epsilon, y, V, store)
    store = populate_store(A, epsilon, y, V, store);

    n = size(A,1);
    m = floor((2*(n-1))/2)+1;
    
    M = zeros(m+2,3*n);
    W = zeros(size(V) + [0,4]);
    W(:,3:end-2) = V;

    M(1:m+2,:) = [W(:,3:m+4).' W(:,2:m+3).' W(:,1:m+2).'];
    
    E = -(M'/(M*M'+epsilon*eye(m+2)))*(M*A.');
    E = E.';

    prod = (A + E)*M.';
end

% function E = recover_exact(A, V, tol)
%     n = size(A,1);
%     m = floor((2*(n-1))/2)+1;
% 
%     M = zeros(m+2,3*n);
%     W = zeros(size(V) + [0,4]);
%     W(:,3:end-2) = V;
% 
%     M(1:m+2,:) = [W(:,3:m+4).' W(:,2:m+3).' W(:,1:m+2).'];
% 
%     E = (A*M.')*pinv(M.',tol);
% 
% end


end
