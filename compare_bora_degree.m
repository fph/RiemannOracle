addpath('./bora-das/BFGS/m-files for dist to singularity/F norm')
addpath('./bora-das/BFGS/hanso3_0/hanso3_0')

rng(1)

options = struct();
options.tolgradnorm = 1e-6;
options.solver = @trustregions;
options.verbosity = 1;
options.epsilon_decrease = 'f';
options.max_outer_iterations = 800;

use_hessian = true;

list_degrees = 2:6;
n_sample = 100;

d_riemann = zeros(list_degrees(end), n_sample);
d_bora = zeros(list_degrees(end), n_sample);

t_riemann = zeros(list_degrees(end), n_sample);
t_bora = zeros(list_degrees(end), n_sample);

m = 15;
for k = list_degrees
    options.maxiter = round(sqrt(k)*40);
    for j = 1:n_sample
        j
        A = randn(m,m*(k+1)) + 1i*randn(m,m*(k+1));
    
        d = floor((k*(m-1))/2);
        V0 = randn(m,d+1);
        V0 = V0./norm(V0,'f');
          
        tic
        [initial_vector,optimal_vector,optimize_at,minimum_distance]=dist_sing_BFGS_F(A);
        t1 = toc;
    
        options.stopping_criterion = norm((A - A*optimize_at*pinv(optimize_at))*optimize_at,'f');
    
        tic
        % Right kernel:
        problem = nearest_singular_polynomial(A, d, use_hessian);
    
        [V_right,~,info_right] = penalty_method(problem, V0, options);
    
        % Left kernel:
        A = reshape(A,m,m,k+1);
        A = pagetranspose(A);
        A = reshape(A,m,m*(k+1));
    
        problem = nearest_singular_polynomial(A, d, use_hessian);
    
        [V_left,~,info_left] = penalty_method(problem, V0, options);
        
        t2 = toc;

        % Choose the smaller one
        d_riemann(k,j) = min(norm(info_left.Delta,'fro'), norm(info_right.Delta,'fro'));
        d_bora(k,j) = minimum_distance;
    
        t_riemann(k,j) = t2;
        t_bora(k,j) = t1;

        % save(['bora_','degrees2to6_new'])
    end
end


% Total time spent (in minutes)
(sum(t_bora,'all') + sum(t_riemann,'all')) / 60

d_means = [mean(d_riemann(list_degrees, :),2) mean(d_bora(list_degrees, :),2)];
d_medians = [median(d_riemann(list_degrees, :),2) median(d_bora(list_degrees, :),2)];

t_means = [mean(t_riemann(list_degrees, :),2) mean(t_bora(list_degrees, :),2)];
t_medians = [median(t_riemann(list_degrees, :),2) median(t_bora(list_degrees, :),2)];

% 
figure;
plot(list_degrees,d_medians,'--x')
xlabel('degree','FontSize',14)
ylabel('distance','FontSize',14)
h = legend('Oracle (median)','Das-Bora (median)');
set(h, 'Location', 'NorthWest')

figure;
plot(list_degrees,d_means,'--x')
% extraInputs = {'interpreter','latex','fontsize',14}; % name, value pairs
xlabel('degree','FontSize',14)
ylabel('distance','FontSize',14)
h = legend('Oracle (mean)','Das-Bora (mean)');
set(h, 'Location', 'NorthWest')

figure;
plot(list_degrees,t_medians,'--x')
xlabel('degree','FontSize',14)
ylabel('time (s)','FontSize',14)
h = legend('Oracle (median)','Das-Bora (median)');
set(h, 'Location', 'NorthWest')

figure;
plot(list_degrees,t_means,'--x')
% extraInputs = {'interpreter','latex','fontsize',14}; % name, value pairs
xlabel('degree','FontSize',14)
ylabel('time (s)','FontSize',14)
h = legend('Oracle (mean)','Das-Bora (mean)');
set(h, 'Location', 'NorthWest')
% 
ties = sum(abs(d_riemann(list_degrees,:) - d_bora(list_degrees,:)) < 1e-8, 2);
we_win = sum(d_riemann(list_degrees,:) < d_bora(list_degrees,:) - 1e-8,2);
they_win = sum(d_riemann(list_degrees,:) > d_bora(list_degrees,:) + 1e-8,2);

figure;
plot(list_degrees,[we_win, they_win, ties],'--x')
% extraInputs = {'interpreter','latex','fontsize',14}; % name, value pairs
xlabel('degree','FontSize',14)
ylabel('frequency','FontSize',14)
h = legend('Oracle wins','Das-Bora wins','Ties');
set(h, 'Location', 'NorthWest')

% save(['bora_','degrees2to6_new'])

