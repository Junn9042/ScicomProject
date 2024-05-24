% Define the objective function
f = @(x) x^4 - 4*x^3 + x^2 + 9*x;

% Set initial guess, maximum number of iterations, and tolerance
initial_point = 5;
target_point = 1;
max_iterations = 1000;
tol = 1e-6;

% Call the newton function to find the minimum
[best_solution, best_cost, elapsed_time] = newton_method(f, initial_point,target_point, max_iterations, tol);

% Display the results
disp('Best Solution:');
disp(best_solution);

disp('Best Cost:');
disp(best_cost);

disp('Elapsed Time:');
disp(elapsed_time);