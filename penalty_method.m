function [x cost info] = penalty_method(problem, x0, options)
% x = penalty_method(problem, x0, options)
%
% x0 may be empty, as with other Manopt functions.
%
% Solve a structured distance problem with the penalty method.
% If options.y is not empty, it is used as the starting value for the 
% dual variable in an augmented Lagrangian method.

    default.starting_epsilon = 1;
    default.epsilon_decrease = 0.5;
    default.max_outer_iterations = 40;
    default.y = [];

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
    while(true)
        k = k + 1;
        regproblem = apply_regularization(problem, epsilon, y);
        
        fprintf("Solving: outer iter=%d, epsilon=%e, norm(y)=%e...\n", k, epsilon, norm(y));
        [x, cost, info] = manoptsolve(regproblem, x, options);
        
        [cons, store] = regproblem.constraint(x, struct());
        orig_cost = problem.cost(x, struct()); % we cannot reuse store because it's a different problem
        relative_constraint_error = norm(cons) / store.normAv;
        fprintf("Solved in %d solver step(s). Cost = %e, non-regularized cost = %e, relative constraint error: %e.\n", ...
            length(info), cost, orig_cost, relative_constraint_error);

        % we want to break out here so we return the y truly used
        if relative_constraint_error < 1e-16 || k == options.max_outer_iterations
            break
        end

        if not(isempty(options.y))
            y = y + 1/epsilon * cons;
        end
        
        if isempty(options.epsilon_decrease)
            % adaptive decrease
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
            epsilon = epsilon * current_epsilon_decrease
        else
            epsilon = epsilon * options.epsilon_decrease;
        end    
    end
    Delta = regproblem.minimizer(x, struct());

    info = struct();
    info.y = y;
    info.last_epsilon = epsilon;
    info.Delta = Delta;
end
