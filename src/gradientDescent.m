function [xmin, fmin, elapsedTime] = gradientDescent(f, initial_point,target_point, alpha_init, max_iter, tol, epsilon)
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
    
    dim = length(initial_point);

    % Khởi tạo biến symbolic
    x = sym('x', [1, dim]); 
    % Tính gradient của hàm mục tiêu
    gradf = gradient(f(x), x);

    % Chuyển hàm symbolic gradient thành hàm số
    grad_f = matlabFunction(gradf, 'Vars', {x});

    x = initial_point; % Ensure x is a row vector
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
    plot(path(:,1), path(:,2), 'r-o', 'LineWidth', 2, 'MarkerFaceColor', 'r');

    plot(target_point(1), target_point(2), 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'g');

    xlabel('x1');
    ylabel('x2');
    title('Contour plot of f(x) with solution path');
    legend('Contours', 'Solution path');
    hold off;

    
end

