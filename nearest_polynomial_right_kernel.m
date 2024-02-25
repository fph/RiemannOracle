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

for j=1:m+2
    M(j,:) = [W(:,j+2).' W(:,j+1).' W(:,j).'];
end

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

for j=1:m+2
    M(j,:) = [W(:,j+2).' W(:,j+1).' W(:,j).'];
end

% f = norm(((M'/(M*M'+eps_reg*eye(m+2)))*(M*A.')),'f')^2;
f = real(trace((A.')'*((M'/(M*M'+eps_reg*eye(m+2)))*(M*A.'))));

end

function g = egrad(V)

    M = zeros(m+2,3*n);
    W = zeros(size(V) + [0,4]);
    W(:,3:end-2) = V;

    for j=1:m+2
        M(j,:) = [W(:,j+2).' W(:,j+1).' W(:,j).'];
    end
    
    A = [A0 A1 A2];
    M_reg = M*M'+eps_reg*eye(m+2);
    M_reg_inv = (M_reg^0) / M_reg;
    x = (M'/(M*M'+eps_reg*eye(m+2)))*(M*A.');
    
    grad_M = 2*(M_reg_inv*M*A.'*(A.'-x)');

    
    g = zeros(size(V));
    for j=1:m
        g(:,j) = grad_M(j,1:n) + grad_M(j+1,n+1:2*n) + grad_M(j+2,2*n+1:3*n);
    end

end


function H = ehess(V, dV)
    
    M = zeros(m+2,3*n);
    W = zeros(size(V) + [0,4]);
    W(:,3:end-2) = V;
    dW = zeros(size(V) + [0,4]);
    dW(:,3:end-2) = dV;
    
    for j=1:m+2
        M(j,:) = [W(:,j+2).' W(:,j+1).' W(:,j).'];
        d(j,:) = [dW(:,j+2).' dW(:,j+1).' dW(:,j).'];
    end
    
    A = [A0 A1 A2];
    M_reg = M*M'+eps_reg*eye(m+2);
    M_reg_inv = (M_reg^0) / M_reg;
    
    D_inv = - (M_reg_inv*(d*M'+M*d'))*M_reg_inv; 
    
    x = (M'/(M*M'+eps_reg*eye(m+2)))*(M*A.');
       
    term1 = D_inv*M*A.'*(A.'-x)';
    term2 = M_reg_inv*d*A.'*(A.'-x)';
    term3 = - M_reg_inv*M*A.'*(d'*M_reg_inv*M*A.' + M'*D_inv*M*A.' + M'*M_reg_inv*d*A.')';

    H_M = 2*(term1 + term2 + term3);
    
    H = zeros(size(V));
    for j=1:m
        H(:,j) = H_M(j,1:n) + H_M(j+1,n+1:2*n) + H_M(j+2,2*n+1:3*n);
    end
        
end


function L_dotdot = compute_L_dotdot(M, M_reg_inv, eps_reg, j1, k1, j2, k2)

    M_dot1 = zeros(m+2,3*n);
    for i = 1:3
        M_dot1(j1+i,k1+(i-1)*n) = 1;
    end
    M_dot2 = zeros(m+2,3*n);
    for i = 1:3
        M_dot2(j2+i,k2+(i-1)*n) = 1;
    end
    
    D_inv1 = compute_D_inv(M, M_reg_inv, eps_reg, j1, k1);
    D_inv2 = compute_D_inv(M, M_reg_inv, eps_reg, j2, k2);
    
    L_dotdot = M_dot1'*D_inv2 + M_dot2'*D_inv1 - ...
        M'*D_inv2*(M_dot1*M' + M*M_dot1')*M_reg_inv  - ... 
        M'*M_reg_inv*(M_dot1*M_dot2' + M_dot2*M_dot1')*M_reg_inv  - ...
        M'*M_reg_inv*(M_dot1*M' + M*M_dot1')*D_inv2;
   
end

function L_dotdot_re_im = compute_L_dotdot_re_im(M, M_reg_inv, eps_reg, j1, k1, j2, k2)

    M_dot1 = zeros(m+2,3*n);
    for i = 1:3
        M_dot1(j1+i,k1+(i-1)*n) = 1;
    end
    M_dot2 = zeros(m+2,3*n);
    for i = 1:3
        M_dot2(j2+i,k2+(i-1)*n) = 1;
    end
    
    D_inv1 = compute_D_inv(M, M_reg_inv, eps_reg, j1, k1);
    D_inv2_im = compute_D_inv_im(M, M_reg_inv, eps_reg, j2, k2);
    
    L_dotdot_re_im = M_dot1'*D_inv2_im - M_dot2'*D_inv1 - ...
        M'*D_inv2_im*(M_dot1*M' + M*M_dot1')*M_reg_inv  - ... 
        M'*M_reg_inv *(-M_dot1*M_dot2' + M_dot2*M_dot1')*M_reg_inv  - ...
        M'*M_reg_inv *(M_dot1*M' + M*M_dot1')*D_inv2_im;
   
end


function L_dotdot_im_im = compute_L_dotdot_im_im(M, M_reg_inv, eps_reg, j1, k1, j2, k2)

    M_dot1 = zeros(m+2,3*n);
    for i = 1:3
        M_dot1(j1+i,k1+(i-1)*n) = 1;
    end
    M_dot2 = zeros(m+2,3*n);
    for i = 1:3
        M_dot2(j2+i,k2+(i-1)*n) = 1;
    end
    
    D_inv1_im = compute_D_inv_im(M, M_reg_inv, eps_reg, j1, k1);
    D_inv2_im = compute_D_inv_im(M, M_reg_inv, eps_reg, j2, k2);
    
    L_dotdot_im_im = -M_dot1'*D_inv2_im - M_dot2'*D_inv1_im - ...
        M'*D_inv2_im*(M_dot1*M' - M*M_dot1')*M_reg_inv  - ... 
        M'*M_reg_inv *(-M_dot1*M_dot2' - M_dot2*M_dot1')*M_reg_inv  - ...
        M'*M_reg_inv *(M_dot1*M' - M*M_dot1')*D_inv2_im;
   
end



function D_inv = compute_D_inv(M, M_reg_inv, eps_reg, j, k)

    M_dot = zeros(m+2,3*n);
    for i = 1:3
        M_dot(j+i,k+(i-1)*n) = 1;
    end
    D_inv = - (M_reg_inv*(M_dot*M'+M*M_dot'))*M_reg_inv ;
    
end

function D_inv_im = compute_D_inv_im(M, M_reg_inv, eps_reg, j, k)

    M_dot = zeros(m+2,3*n);
    for i = 1:3
        M_dot(j+i,k+(i-1)*n) = 1;
    end
    D_inv_im = - (M_reg_inv*(M_dot*M'-M*M_dot'))*M_reg_inv;
    
end

end
