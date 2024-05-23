function [xmin, fmin, iter, elapsedTime] = gradientDescent(f, grad_f, x0, alpha_init, max_iter, tol, epsilon)
    % Inputs:
    %   f - function handle to the objective function
    %   grad_f - function handle to the gradient of the objective function
    %   x0 - initial point (row vector)
    %   alpha_init - initial learning rate
    %   max_iter - maximum number of iterations
    %   tol - tolerance for the stopping criterion
    %   epsilon - parameter for the stopping criterion in the backtracking line search
    
    % Outputs:
    %   xmin - the point where the minimum is attained
    %   fmin - the minimum value of the function
    %   iter - number of iterations performed
    %   elapsedTime - time taken to run the algorithm

    tic; % start timing
    x = x0(:)'; % Ensure x is a row vector
    iter = 0;
    path = x; % Record the path

    while iter < max_iter
        grad = grad_f(x)'; % Ensure gradient is a row vector
        alpha = alpha_init;
        while true
            u = x - alpha * grad;
            if f(u) <= f(x) - epsilon * alpha * norm(grad)^2
                break;
            else
                alpha = alpha / 2;
            end
        end

        x_new = x - alpha * grad;
        path = [path; x_new];

        if norm(x_new - x, 2) < tol || norm(grad, 2) < tol
            break;
        end

        x = x_new;
        iter = iter + 1;
    end

    elapsedTime = toc; % end timing
    xmin = x;
    fmin = f(x);

    % Plot contour of the function and solution path
    figure;
    % Generate grid for contour plot
    x1_vals = linspace(min(path(:,1)) - 1, max(path(:,1)) + 1, 100);
    x2_vals = linspace(min(path(:,2)) - 1, max(path(:,2)) + 1, 100);
    [X1, X2] = meshgrid(x1_vals, x2_vals);
    F = arrayfun(@(x1, x2) f([x1; x2]), X1, X2);
    contour(X1, X2, F, 50);
    hold on;
    plot(path(:,1), path(:,2), '-o', 'LineWidth', 2, 'MarkerSize', 5);
    xlabel('x1');
    ylabel('x2');
    title('Contour plot of f(x) with solution path');
    legend('Contours', 'Solution path');
    hold off;
end

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
