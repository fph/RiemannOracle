function penalty_method(problem, x0, options)
    default.starting_epsilon = 1;
    default.epsilon_decrease = 0.5;
    default.outer_iterations = 20;
    default.use_y = false;

    options = mergeOptions(default, options);

    epsilon = options.starting_epsilon;
    x = x0;
    if options.use_y
        y = 0;
    end

    for k = 1:options.outer_iterations
        regproblem = apply_regularization(problem, epsilon, 0);
        x = manoptsolve(regproblem, x, options);
        if options.use_y
            y = y + TODO;
        end
        epsilon = epsilon * options.epsilon_decrease;
    end
end
