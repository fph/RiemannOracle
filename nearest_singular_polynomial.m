function [D0,D1,D2,e,t,V,infotable] = nearest_singular_polynomial(A0, A1, A2, maxiter, timemax, V0)

V0_r = V0;
V0_l = V0;

n = length(A1);
m = floor((2*(n-1))/2)+1;

A = [A0 A1 A2];

M = zeros(m+2,3*n);
W = zeros(size(V0) + [0,4]);
W(:,3:end-2) = V0;

M(1:m+2,:) = [W(:,3:m+4).' W(:,2:m+3).' W(:,1:m+2).'];

eps_reg = 1;
for i = 1:100
    [D0_r,D1_r,D2_r,e_r,t_r,V_r,infotable_r] = nearest_polynomial_right_kernel(A0, A1, A2, maxiter, timemax, V0_r, eps_reg);
    V0_r = V_r;
    eps_reg = eps_reg*1e-2;
    W = zeros(size(V_r) + [0,4]);
    W(:,3:end-2) = V_r;

    M(1:m+2,:) = [W(:,3:m+4).' W(:,2:m+3).' W(:,1:m+2).'];
    f = real(trace((A.')'*((M'/(M*M'+eps_reg*eye(m+2)))*(M*A.'))));
    
    while f > 2.5*e_r(end)^2
        eps_reg = 1.1*eps_reg;
        f = real(trace((A.')'*((M'/(M*M'+eps_reg*eye(m+2)))*(M*A.'))));
    end
    if eps_reg < 1e-13
        break;
    end

end

W(:,3:end-2) = V_r;
M(1:m+2,:) = [W(:,3:m+4).' W(:,2:m+3).' W(:,1:m+2).'];


A = [A0.' A1.' A2.'];
eps_reg = 1;
for i = 1:100
    [D0_l,D1_l,D2_l,e_l,t_l,V_l,infotable_l] = nearest_polynomial_right_kernel(A0.', A1.', A2.', maxiter, timemax, V0_l, eps_reg);
    V0_l = V_l;

    eps_reg = eps_reg*1e-2;
    W = zeros(size(V_l) + [0,4]);
    W(:,3:end-2) = V_l;

    M(1:m+2,:) = [W(:,3:m+4).' W(:,2:m+3).' W(:,1:m+2).'];
    f = real(trace((A.')'*((M'/(M*M'+eps_reg*eye(m+2)))*(M*A.'))));
    
    while f > 2.5*e_l(end)^2
        eps_reg = 1.1*eps_reg;
        f = real(trace((A.')'*((M'/(M*M'+eps_reg*eye(m+2)))*(M*A.'))));
    end

    if eps_reg < 1e-13
        break;
    end
    
end


if e_r(end) < e_l(end)

    D0 = D0_r;
    D1 = D1_r;
    D2 = D2_r;
    e = e_r;
    t = [t_r; t_l+t_r(end)];
    V = V_r;
    infotable = infotable_r;

else
    
    D0 = D0_l.';
    D1 = D1_l.';
    D2 = D2_l.';
    e = e_l;
    t = [t_r; t_l+t_r(end)];
    V = V_l;
    infotable = infotable_l;

    
end
    
end
