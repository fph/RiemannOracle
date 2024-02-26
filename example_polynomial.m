clear all;

% A0 = [13 2 1; 2 7 2; 1 2 4];
% A1 = [0 -2 4; 2 0 -2; -4 2 0];
% A2 = eye(3,3);

rng(5)
m = 5;
A0 = randn(m) + 1i*randn(m);
A1 = randn(m) + 1i*randn(m);
A2 = randn(m) + 1i*randn(m);

% A0 = randn(m);
% A1 = randn(m);
% A2 = randn(m);


n = length(A1);

V0 = randn(n,floor((2*(n-1))/2)+1);
V0 = V0./norm(V0,'f');

maxiter = 5000;
timemax = 10;

% eps_regs = 10.^-[1:1:13];

[D0,D1,D2,e,t,V,infotable] = nearest_singular_polynomial(A0, A1, A2, maxiter, timemax, V0);
          
%     M = block_toepl(A2+D2,A1+D1,A0+D0,2*n-1);
%     svd(M)
%     keyboard
norm([D0 D1 D2],'f')





