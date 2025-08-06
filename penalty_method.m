function [x, cost, info, results] = penalty_method(problem, x0, options)
% x = penalty_method(problem, x0, options)
%
% x0 may be empty, as with other Manopt functions.
%
% Solve a structured distance problem with the penalty method.
%
%
% If options.y is not empty, it is used as the starting value for the 
% dual variable in an augmented Lagrangian method.
% We recommend using options.y = 0 rather than the default [].
%
% options.epsilon_decrease is the strategy to use to reduce the
% regularization paramter epsilon at each iteration. Supported values are:
% a numerical constant (default=0.5), 'f' for an adaptive decrease based on 
% the function value, 'g' for a decrease based on the value of the gradient.
%
% options.stopping_criterion specifies the value that the absolute 
% error in the constraint needs to reach before the iteration is stopped.
% If no value is provided, then a relative error of 1e-16 is used as the
% stopping cirerion.
%
% Note for those implementing new problem structures:
% This function accesses store.normAv and store.condM, which must exist.

    default.starting_epsilon = 1;
    default.epsilon_decrease = 0.5;
    default.max_outer_iterations = 40;
    default.y = [];
    default.stopping_criterion = [];

    if not(exist('options', 'var'))
        options = struct();
    end
    if not(exist('x0', 'var'))
        x0 = [];
    end

    options = mergeOptions(default, options);

    epsilon = options.starting_epsilon;
    x = x0;
    if not(isempty(options.y))
        y = options.y;
    else
        y = 0;
    end
    k = 0;
    results = table();
    saved_warning_state = warning();
    warning('off', 'MATLAB:table:RowsAddedExistingVars'); % to fill the table more easily

    while(true)
        k = k + 1;        
        regproblem = apply_regularization(problem, epsilon, y);
        
        fprintf("Solving: outer iter=%d, epsilon=%e, norm(y)=%e...\n", k, epsilon, norm(y));
        [x, cost, info] = manoptsolve(regproblem, x, options);

        results.iteration(k) = k;
        results.epsilon(k) = epsilon;
        results.minfeps(k) = cost;
        results.normy(k) = norm(y);
        
        [cons, store] = regproblem.constraint(x, struct());
        orig_cost = problem.cost(x, struct()); % we cannot reuse store because it's a different problem
        relative_constraint_error = norm(cons) / store.normAv;
        fprintf("Solved in %d solver step(s). Cost = %e, non-regularized cost = %e, cond(M) = %e, relative constraint error: %e.\n", ...
            length(info), cost, orig_cost, store.condM, relative_constraint_error);

        results.inner_its(k) = length(info);
        results.fx(k) = orig_cost;
        results.relative_constraint_error(k) = relative_constraint_error;
        results.condM(k) = store.condM;
        if not(isempty(options.y))
            results.augmented_lagrangian(k) = cost - epsilon*norm(y)^2;
        end

        if isempty(options.stopping_criterion)
            constraint_satisfied = relative_constraint_error < 1e-16;
        else
            constraint_satisfied = norm(cons) < options.stopping_criterion;
        end

        % we want to break out here so that we can return the y truly used
        % before updates    
        if constraint_satisfied || k == options.max_outer_iterations
            break
        end

        if not(isempty(options.y))
            y = y + 1/epsilon * cons;
        end
        
        switch options.epsilon_decrease
            case 'g' % adaptive decrease based on gradient value
                current_epsilon_decrease = 0.5;
                eg = problem.genegrad(epsilon * current_epsilon_decrease, y, x, struct());
                rg = problem.M.egrad2rgrad(x, eg);
                while norm(rg) > 1e-4
                    current_epsilon_decrease = (1 + current_epsilon_decrease) / 2;
                    eg = problem.genegrad(epsilon * current_epsilon_decrease, y, x, struct());
                    rg = problem.M.egrad2rgrad(x, eg);
                    if current_epsilon_decrease > 0.95
                        break
                    end
                end
                epsilon = epsilon * current_epsilon_decrease;
            case 'f' % adaptive decrease based on function value                
                current_epsilon_decrease = 1e-2; % "aggressive" parameters based on the polynomial code and experiments
                fval = problem.gencost(epsilon * current_epsilon_decrease, y, x, struct());
                while fval > 2.5 * cost                     
                    current_epsilon_decrease = current_epsilon_decrease * 1.1;
                    fval = problem.gencost(epsilon * current_epsilon_decrease, y, x, struct());
                    if current_epsilon_decrease > 0.95
                        break
                    end
                end
                epsilon = epsilon * current_epsilon_decrease;
            otherwise  % fixed numerical value
                assert(isscalar(options.epsilon_decrease));
                epsilon = epsilon * options.epsilon_decrease;
        end    
    end
    [Delta AplusDelta last_store] = regproblem.minimizer(x, struct());

    info = struct();
    info.y = y;
    info.last_epsilon = epsilon;
    info.Delta = Delta;
    info.last_store = last_store;

    warning(saved_warning_state); % restoring the previous state
end
