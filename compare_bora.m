addpath('./bora-das/BFGS/m-files for dist to singularity/F norm')
addpath('./bora-das/BFGS/hanso3_0/hanso3_0')
% addpath('..')  
% addpath('../manopt/manopt')
% addpath(genpath('../manopt/manopt/manopt'))

% clear all;

rng(5)
% m = 12;
k = 2;

options = struct();
options.maxiter = 50;
% options.maxtime = 4;
options.tolgradnorm = 1e-6;
% options.debug=0;
options.solver = @trustregions;
options.verbosity = 1;
options.epsilon_decrease = 'f';
options.max_outer_iterations = 800;
% options.y = 0;


use_hessian = true;


list_sizes = [5 10 15 20];
n_sample = 1;

d_riemann = zeros(list_sizes(end), n_sample);
d_bora = zeros(list_sizes(end), n_sample);

t_riemann = zeros(list_sizes(end), n_sample);
t_bora = zeros(list_sizes(end), n_sample);


for m = list_sizes
    for j = 1:n_sample
        
        A0 = randn(m) + 1i*randn(m);
        A1 = randn(m) + 1i*randn(m);
        A2 = randn(m) + 1i*randn(m);
        A = [A0 A1 A2];
    
        % A3 = randn(m) + 1i*randn(m);
        % A = [A0 A1 A2 A3];
        % 
        % A = A + info_right.Delta;
    
        n = length(A1);
    
        d = floor((k*(n-1))/2);
    
        V0 = randn(n,d+1);
        V0 = V0./norm(V0,'f');
    
        
        tic
        [initial_vector,optimal_vector,optimize_at,minimum_distance]=dist_sing_BFGS_F(A);
        t1 = toc;
    
        options.stopping_criterion = norm((A - A*optimize_at*pinv(optimize_at))*optimize_at,'f');
    
        % keyboard
        tic
        % Right kernel:
        problem = nearest_singular_polynomial(A, d, use_hessian);
    
        [V_right,~,info_right] = penalty_method(problem, V0, options);
    
        % Left kernel:
        A = [A0.' A1.' A2.'];
        % A = [A0.' A1.' A2.' A3.'];
    
        problem = nearest_singular_polynomial(A, d, use_hessian);
    
        [V_left,~,info_left] = penalty_method(problem, V0, options);
        
        t2 = toc;
        % Choose the smaller one
        d_riemann(m,j) = min(norm(info_left.Delta,'fro'), norm(info_right.Delta,'fro'));
        d_bora(m,j) = minimum_distance;
    
        t_riemann(m,j) = t2;
        t_bora(m,j) = t1;
        % keyboard
    end
end


% d_riemann = d_riemann(list_sizes, :);
% d_bora = d_bora(list_sizes, :);
% 
% t_riemann = t_riemann(list_sizes, :);
% t_bora = t_bora(list_sizes, :); 


% save(['bora_','sizes5to20'])

