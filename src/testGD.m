% Example usage:
f = @(x) x(1)^2 + x(2)^2; % Objective function
initial_point = [5, 5]; % Initial point (row vector)
target_point = [0, 0];
alpha = 0.1; % Initial learning rate
beta = 0.05;
max_iter = 1000; % Maximum number of iterations
tol = 1e-6; % Tolerance for stopping criterion
epsilon = 0.5; % Epsilon parameter for backtracking line search

%[xmin, fmin, elapsedTime] = gradient_descent(f, initial_point,target_point, alpha, max_iter, tol, epsilon);

[xmin, fmin, elapsedTime] = gradient_descent_with_momentum(f, initial_point,target_point, alpha, beta, max_iter, tol);

fprintf('Global minimum: [%f, %f]\n', xmin(1), xmin(2));
fprintf('Function value at minimum: %f\n', fmin);
fprintf('Elapsed time: %f seconds\n', elapsedTime);