% Example usage:
f = @(x)(1 + (x(1) + x(2) + 1)^2 * (19 - 14*x(1) + 3*x(1)^2 - 14*x(2) + 6*x(1)*x(2) + 3*x(2)^2)) * (30 + (2*x(1) - 3*x(2))^2 * (18 - 32*x(1) + 12*x(1)^2 + 48*x(2) - 36*x(1)*x(2) + 27*x(2)^2));
initial_point = [-1, 1]; % Initial point (row vector)
target_point = [0, -1];
alpha = 0.001; % Initial learning rate
beta = 0.05;
max_iter = 5; % Maximum number of iterations
tol = 1e-6; % Tolerance for stopping criterion
epsilon = 0.5; % Epsilon parameter for backtracking line search

%[xmin, fmin, elapsedTime] = gradient_descent(f, initial_point,target_point, alpha, max_iter, tol, epsilon);

[xmin, fmin, elapsedTime] = gradient_descent_with_momentum(f, initial_point,target_point, alpha, beta, max_iter, tol);

fprintf('Global minimum: [%f, %f]\n', xmin(1), xmin(2));
fprintf('Function value at minimum: %f\n', fmin);
fprintf('Elapsed time: %f seconds\n', elapsedTime);