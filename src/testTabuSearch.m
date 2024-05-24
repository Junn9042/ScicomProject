
% Objective function
objective_function = @(x) x^4 - 4*x^3 + x^2 + 9*x;

gp = @(x)(1 + (x(1) + x(2) + 1)^2 * (19 - 14*x(1) + 3*x(1)^2 - 14*x(2) + 6*x(1)*x(2) + 3*x(2)^2)) * (30 + (2*x(1) - 3*x(2))^2 * (18 - 32*x(1) + 12*x(1)^2 + 48*x(2) - 36*x(1)*x(2) + 27*x(2)^2));


% Parameters
max_iterations = 10000;       % Number of iterations
neighborhood_size = 10;     % Number of neighbors to explore
tabu_list_size = 5;         % Size of the tabu list
initial_point = [5, 5];  % Initial solution (2-dimensional point)
target_point = [0, -1];

% Run Tabu Search
[best_solution, best_cost, elapsed_time] = tabu_search(gp,initial_point, target_point, max_iterations, neighborhood_size, tabu_list_size);

% Display final result
disp('Best Solution:');
disp(best_solution);
disp('Best Cost:');
disp(best_cost);
disp('Elapsed Time:');
disp(elapsed_time);

