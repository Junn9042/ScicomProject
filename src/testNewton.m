% Define the objective function
f = @(x)(1 + (x(1) + x(2) + 1)^2 * (19 - 14*x(1) + 3*x(1)^2 - 14*x(2) + 6*x(1)*x(2) + 3*x(2)^2)) * (30 + (2*x(1) - 3*x(2))^2 * (18 - 32*x(1) + 12*x(1)^2 + 48*x(2) - 36*x(1)*x(2) + 27*x(2)^2));


% Set initial guess, maximum number of iterations, and tolerance
initial_point = [3, 3];
target_point = [0, -1];
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