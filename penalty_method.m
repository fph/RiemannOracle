function x = penalty_method(problem, x0, options)
% x = penalty_method(problem, x0, options)
%
% x0 may be empty, as with other Manopt functions.
%
% Solve a structured distance problem with the penalty method.
% If options.y is not empty, it is used as the starting value for the 
% dual variable in an augmented Lagrangian method.

    default.starting_epsilon = 1;
    default.epsilon_decrease = 0.5;
    default.outer_iterations = 20;
    default.y = [];

    options = mergeOptions(default, options);

    epsilon = options.starting_epsilon;
    x = x0;
    if not(isempty(options.y))
        y = options.y;
    else
        y = 0;
    end

    for k = 1:options.outer_iterations
        regproblem = apply_regularization(problem, epsilon, y);
        
        fprintf("Solving: iter=%d, epsilon=%e, norm(y)=%e...\n", k, epsilon, norm(y));
        [x, cost, info] = manoptsolve(regproblem, x, options);
        
        cons = regproblem.constraint(x, struct());
        orig_cost = problem.cost(x, struct());
        fprintf("Solved in %d solver step(s). Cost = %e, non-regularized cost = %e, constraint norm: %e.\n", ...
            length(info), cost, orig_cost, norm(cons));
        if not(isempty(options.y))
            y = y + 1/epsilon * cons;
        end
        epsilon = epsilon * options.epsilon_decrease;      
    end
end
