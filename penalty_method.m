function x = penalty_method(problem, x0, options)
%  penalty_method(problem, x0, options)
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
        fprintf("Solving: iter=%d, epsilon=%g, norm(y)=%g\n", k, epsilon, norm(y));
        x = manoptsolve(regproblem, x, options);
        cons = regproblem.constraint(x, struct());
        fprintf("Constraint norm: %g\n", norm(cons));
        if not(isempty(options.y))
            y = y + 1/epsilon * cons;
        end
        epsilon = epsilon * options.epsilon_decrease;      
    end
end