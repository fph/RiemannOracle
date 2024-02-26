function [D0,D1,D2,e,t,V,infotable] = nearest_polynomial_right_kernel(A0, A1, A2, maxiter, timemax, V0, eps_reg)

n = length(A1);
m = floor((2*(n-1))/2)+1;

% problem.M = euclideancomplexfactory(n, 2*n-1);
problem.M = spherecomplexfactory(n, m);

problem.cost = @cost;

% Euclidean gradient. Projection from ambient space to the tangent space of
% U(n) is handled automatically (see stiefelcomplexfactory documentation)
problem.egrad = @egrad;
% Euclidean Hessian. Projection is handled automatically.
problem.ehess = @ehess;

options.maxiter = maxiter;
options.maxtime = timemax;
options.tolgradnorm = 1e-6;
options.Delta_bar = 4.47214*1e-0;
options.Delta0 = options.Delta_bar/8;
options.debug=0;
options.rho_regularization = 1e3;

warning('off', 'manopt:getHessian:approx');

[V, xcost, info, options] = trustregions(problem, V0, options);

infotable = struct2table(info);
e = sqrt(infotable.cost);
t = infotable.time;

A = [A0 A1 A2];

M = zeros(m+2,3*n);
W = zeros(size(V) + [0,4]);
W(:,3:end-2) = V;

M(1:m+2,:) = [W(:,3:m+4).' W(:,2:m+3).' W(:,1:m+2).'];


perturbation = -(M'/(M*M'+eps_reg*eye(m+2)))*(M*A.');
perturbation = perturbation.';

D0 = perturbation(1:n,1:n);
D1 = perturbation(1:n,n+1:2*n);
D2 = perturbation(1:n,2*n+1:3*n);


function f = cost(V)

M = zeros(m+2,3*n);
W = zeros(size(V) + [0,4]);
W(:,3:end-2) = V;
A = [A0 A1 A2];

M(1:m+2,:) = [W(:,3:m+4).' W(:,2:m+3).' W(:,1:m+2).'];

f = real(trace((A.')'*((M'/(M*M'+eps_reg*eye(m+2)))*(M*A.'))));

end

function g = egrad(V)

    M = zeros(m+2,3*n);
    W = zeros(size(V) + [0,4]);
    W(:,3:end-2) = V;

    M(1:m+2,:) = [W(:,3:m+4).' W(:,2:m+3).' W(:,1:m+2).'];
    
    A = [A0 A1 A2];
    M_reg = M*M'+eps_reg*eye(m+2);
    M_reg_inv = (M_reg^0) / M_reg;
    x = (M'/(M*M'+eps_reg*eye(m+2)))*(M*A.');
    
    grad_M = 2*(M_reg_inv*M*A.'*(A.'-x)');
    g = (grad_M(1:m,1:n) + grad_M(2:m+1,n+1:2*n) + grad_M(3:m+2,2*n+1:3*n)).';

end


function H = ehess(V, dV)
    
    M = zeros(m+2,3*n);
    W = zeros(size(V) + [0,4]);
    W(:,3:end-2) = V;
    dW = zeros(size(V) + [0,4]);
    dW(:,3:end-2) = dV;
    
    M(1:m+2,:) = [W(:,3:m+4).' W(:,2:m+3).' W(:,1:m+2).'];
    d(1:m+2,:) = [dW(:,3:m+4).' dW(:,2:m+3).' dW(:,1:m+2).'];

    A = [A0 A1 A2];
    M_reg = M*M'+eps_reg*eye(m+2);
    M_reg_inv = (M_reg^0) / M_reg;
    
    D_inv = - (M_reg_inv*(d*M'+M*d'))*M_reg_inv; 
    
    x = (M'/(M*M'+eps_reg*eye(m+2)))*(M*A.');
       
    term1 = D_inv*M*A.'*(A.'-x)';
    term2 = M_reg_inv*d*A.'*(A.'-x)';
    term3 = - M_reg_inv*M*A.'*(d'*M_reg_inv*M*A.' + M'*D_inv*M*A.' + M'*M_reg_inv*d*A.')';

    H_M = 2*(term1 + term2 + term3);
    
    H = (H_M(1:m,1:n) + H_M(2:m+1,n+1:2*n) + H_M(3:m+2,2*n+1:3*n)).';

end


end
