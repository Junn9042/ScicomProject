% Example usage:
% Define the objective function
f = @(x) 100*(x(2)-x(1)^2)^2 + (1-x(1))^2;  % Example: simple quadratic function

% Initial guess
x0 = [-1; -1];

% Parameters
max_iter = 1000;
tol = 1e-6;
target_point = [1, 1];
% Call the Quasi-Newton BFGS method
[x_min, fval, iter, x_history] = quasi_newton_bfgs(f, x0, tol, max_iter, target_point);

function [x, fval, iter, x_history] = quasi_newton_bfgs(f, x0, tol, max_iter, target_point)
    % Quasi-Newton method using BFGS update
    % Input:
    % func - handle to the objective function
    % x0 - initial guess
    % tol - tolerance for convergence
    % max_iter - maximum number of iterations
    % Output:
    % x - the found minimum point
    % fval - function value at the minimum point
    % iter - number of iterations performed
    % x_history - history of x values during optimization
    
    x = x0;
    n = length(x);
    H = eye(n);  % Initial approximation to the Hessian is the identity matrix
    iter = 0;
    x_history = x0;
    x1_history = x0(1);
    x2_history = x0(2);

    for k = 1:max_iter
        g = numerical_gradient(f, x);  % Gradient at current point
        if norm(g) < tol
            break;
        end
        p = -H * g;  % Search direction

        % Line search (using backtracking line search)
        alpha = 1;
        rho = 0.9;
        c = 1e-4;
        while f(x + alpha * p) > f(x) + c * alpha * g' * p
            alpha = rho * alpha;
        end

        x_new = x + alpha * p;  % Update the point
        s = x_new - x;  % Step size
        y = numerical_gradient(f, x_new) - g;  % Change in gradient

        % BFGS update
        rho_k = 1 / (y' * s);
        V = eye(n) - rho_k * (s * y');
        H = V' * H * V + rho_k * (s * s');
        
        x = x_new;  % Move to the new point
        iter = iter + 1;
        x_history = [x_history, x];
        
        % Lưu lịch sử giá trị của x1 và x2
        x1_history = [x1_history, x(1)];
        x2_history = [x2_history, x(2)];
    end

    fval = f(x);  % Function value at the found minimum point
    
    % Vẽ đồ thị sự hội tụ và các đường đồng mức
    figure;
    [x1_grid, x2_grid] = meshgrid(-2:0.1:2, -1:0.1:3);
    [X1, X2] = deal(x1_grid, x2_grid); % Sử dụng deal để tách X1 và X2
    % Tính toán f_grid bằng cách lặp qua từng phần tử của X1 và X2
    f_grid = zeros(size(X1));
    for i = 1:numel(X1)
        f_grid(i) = f([X1(i), X2(i)]);
    end

    % Kiểm tra kích thước của f_grid và X1
    size_f_grid = numel(f_grid);
    size_X1 = numel(X1);

    if size_f_grid == size_X1
        f_grid = reshape(f_grid, size(X1));
    else
        error('Kích thước của f_grid và X1 không khớp.');
    end
    contour(x1_grid, x2_grid, f_grid, 50); hold on;
    plot(x1_history, x2_history, 'r-o', 'LineWidth', 2, 'MarkerFaceColor', 'r');
    
    % Vẽ điểm cuối cùng với màu khác
    plot(target_point(1), target_point(2), 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
    
    xlabel('x1');
    ylabel('x2');
    title('Contour plot and convergence path');
    legend('Contours', 'Target point');
    hold off;
    
end

% Hàm tính gradient
function g = numerical_gradient(func, x)
    % Compute gradient using finite difference approximation
    % Input:
    % func - handle to the objective function
    % x - point at which to compute the gradient
    % Output:
    % g - gradient vector (column vector)

    epsilon = 1e-8;  % Small perturbation
    n = length(x);  % Number of variables
    g = zeros(n, 1);  % Initialize gradient vector

    for i = 1:n
        x_forward = x;
        x_backward = x;
        x_forward(i) = x(i) + epsilon;
        x_backward(i) = x(i) - epsilon;
        
        g(i) = (func(x_forward) - func(x_backward)) / (2 * epsilon);
    end
end
