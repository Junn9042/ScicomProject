function [xmin, fmin, elapsedTime] = gradient_descent(f, initial_point,target_point, alpha, max_iterations, tol, epsilon)
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
    alpha_min = 1e-4;

    % Khởi tạo biến symbolic
    x = sym('x', [1, dim]); 
    % Tính gradient của hàm mục tiêu
    gradf = gradient(f(x), x);

    % Chuyển hàm symbolic gradient thành hàm số
    grad_f = matlabFunction(gradf, 'Vars', {x});

    x = initial_point; % Ensure x is a row vector
    iter = 0;
    best_solutions = x; % Record the path

    while iter < max_iterations
        grad = grad_f(x)'; % Ensure gradient is a row vector
        alpha = alpha;
        while true
            u = x - alpha * grad;
            if f(u) <= f(x) - epsilon * alpha * norm(grad)^2
                break;
            else
                alpha = alpha / 2;
            end
        end

        x_new = x - alpha * grad;
        best_solutions = [best_solutions; x_new];

        if norm(x_new - x, 2) < tol || norm(grad, 2) < tol
            break;
        end

        x = x_new;
        iter = iter + 1;
    end

    elapsedTime = toc; % end timing
    xmin = x;
    fmin = f(x);

    if length(initial_point)==2
            % Plot contour of the function and solution path
        figure;
        % Generate grid for contour plot
        [X, Y] = meshgrid(linspace(min(best_solutions(:, 1)) - 1, max(best_solutions(:, 1)) + 1 , 100), linspace(min(best_solutions(:, 2)) - 1, max(best_solutions(:, 2)) + 1, 100));
        Z = arrayfun(@(x, y) f([x, y]), X, Y);
        contour(X, Y, Z, 200);
        hold on;
        plot(best_solutions(:,1), best_solutions(:,2), 'r-o', 'LineWidth', 2, 'MarkerFaceColor', 'r');
    
        plot(target_point(1), target_point(2), 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
    
        xlabel('x1');
        ylabel('x2');
        title('Contour plot of f(x) with solution path - Gradient Descent');
        legend('Contours', 'Solution path');
        hold off;
    end
    
end