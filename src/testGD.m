% Example usage:
f = @(x) x(1)^2 + x(2)^2; % Objective function
grad_f = @(x) [2*x(1); 2*x(2)]; % Gradient of the objective function
x0 = [5, 5]; % Initial point (row vector)
alpha_init = 0.1; % Initial learning rate
max_iter = 1000; % Maximum number of iterations
tol = 1e-6; % Tolerance for stopping criterion
epsilon = 0.5; % Epsilon parameter for backtracking line search

[xmin, fmin, iter, elapsedTime] = gradientDescent(f, grad_f, x0, alpha_init, max_iter, tol, epsilon);

fprintf('Global minimum: [%f, %f]\n', xmin(1), xmin(2));
fprintf('Function value at minimum: %f\n', fmin);
fprintf('Number of iterations: %d\n', iter);
fprintf('Elapsed time: %f seconds\n', elapsedTime);