% Example usage:
f = @(x) 100*(x(2)-x(1)^2)^2 + (1-x(1))^2;
initial_point = [2, 2]; % Initial point (row vector)
target_point = [1, 1];
alpha = 0.001; % Initial learning rate
beta = 0.005;
max_iter = 10000; % Maximum number of iterations
tol = 1e-6; % Tolerance for stopping criterion
epsilon = 0.5; % Epsilon parameter for backtracking line search

%[xmin, fmin, elapsedTime] = gradient_descent(f, initial_point,target_point, alpha, max_iter, tol, epsilon);

[xmin, fmin, elapsedTime] = gradient_descent_with_momentum(f, initial_point,target_point, alpha, beta, max_iter, tol);

fprintf('Global minimum: [%f, %f]\n', xmin(1), xmin(2));
fprintf('Function value at minimum: %f\n', fmin);
fprintf('Elapsed time: %f seconds\n', elapsedTime);