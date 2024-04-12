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


list_sizes = [5 10 15 20 25];
n_sample = 100;

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


% Total time spent (in minutes)
(sum(t_bora,'all') + sum(t_riemann,'all')) / 60

% d_riemann_av = mean(d_riemann(list_sizes, :),2);
% d_bora_av = mean(d_bora(list_sizes, :),2);

% t_riemann_av = mean(t_riemann(list_sizes, :),2);
% t_bora_av = mean(t_bora(list_sizes, :),2); 

d_means = [mean(d_riemann(list_sizes, :),2) mean(d_bora(list_sizes, :),2)];
d_medians = [median(d_riemann(list_sizes, :),2) median(d_bora(list_sizes, :),2)];

t_means = [mean(t_riemann(list_sizes, :),2) mean(t_bora(list_sizes, :),2)];
t_medians = [median(t_riemann(list_sizes, :),2) median(t_bora(list_sizes, :),2)];


% figure;
% plot(list_sizes,d_medians,'--x')
% xlabel('size (n)','FontSize',14)
% ylabel('distance','FontSize',14)
% h = legend('Oracle (median)','Das-Bora (median)');
% set(h, 'Location', 'NorthWest')
% 
% 
% figure;
% plot(list_sizes,d_means,'--x')
% % extraInputs = {'interpreter','latex','fontsize',14}; % name, value pairs
% xlabel('size (n)','FontSize',14)
% ylabel('distance','FontSize',14)
% h = legend('Oracle (mean)','Das-Bora (mean)');
% set(h, 'Location', 'NorthWest')
% 
% figure;
% plot(list_sizes,t_medians,'--x')
% xlabel('size (n)','FontSize',14)
% ylabel('time (s)','FontSize',14)
% h = legend('Oracle (median)','Das-Bora (median)');
% set(h, 'Location', 'NorthWest')
% 
% figure;
% plot(list_sizes,t_means,'--x')
% % extraInputs = {'interpreter','latex','fontsize',14}; % name, value pairs
% xlabel('size (n)','FontSize',14)
% ylabel('time (s)','FontSize',14)
% h = legend('Oracle (mean)','Das-Bora (mean)');
% set(h, 'Location', 'NorthWest')





% save(['bora_','sizes5to25'])

