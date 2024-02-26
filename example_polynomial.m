% An example for the specialised polynomial algorithm
clear all;

rng(5)
m = 5;
A0 = randn(m) + 1i*randn(m);
A1 = randn(m) + 1i*randn(m);
A2 = randn(m) + 1i*randn(m);

n = length(A1);

V0 = randn(n,floor((2*(n-1))/2)+1);
V0 = V0./norm(V0,'f');

maxiter = 5000;
timemax = 10;

[D0,D1,D2,e,t,V,infotable] = nearest_singular_polynomial(A0, A1, A2, maxiter, timemax, V0);
          
norm([D0 D1 D2],'f')





