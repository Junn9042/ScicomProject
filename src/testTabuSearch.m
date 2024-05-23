
% Objective function
objective_function = @(x)(1 + (x(1) + x(2) + 1)^2 * (19 - 14*x(1) + 3*x(1)^2 - 14*x(2) + 6*x(1)*x(2) + 3*x(2)^2)) * (30 + (2*x(1) - 3*x(2))^2 * (18 - 32*x(1) + 12*x(1)^2 + 48*x(2) - 36*x(1)*x(2) + 27*x(2)^2));

% Parameters
max_iterations = 1000;       % Number of iterations
neighborhood_size = 10;     % Number of neighbors to explore
tabu_list_size = 5;         % Size of the tabu list
initial_point = [-3, 5];  % Initial solution (2-dimensional point)
target_point = [0, -1];

% Run Tabu Search
[best_solution, best_cost, running_time] = tabu_search(objective_function, max_iterations, neighborhood_size, tabu_list_size, initial_point, target_point);

% Display final result
fprintf('Best Solution: x = [%f, %f]\n', best_solution(1), best_solution(2));
fprintf('Best Cost: %f\n', best_cost);
fprintf('Running Time: %f seconds\n', running_time);

