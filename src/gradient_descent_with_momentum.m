function [xmin, fmin, elapsedTime] = gradient_descent_with_momentum(f, initial_point, target_point, alpha, beta, max_iter, tol)
    % Inputs:
    %   f - function handle to the objective function
    %   grad_f - function handle to the gradient of the objective function
    %   x0 - initial point (row vector)
    %   alpha - learning rate
    %   beta - momentum coefficient
    %   max_iter - maximum number of iterations
    %   tol - tolerance for the stopping criterion
    
    % Outputs:
    %   xmin - the point where the minimum is attained
    %   fmin - the minimum value of the function
    %   iter - number of iterations performed
    %   elapsedTime - time taken to run the algorithm

    tic; % start timing
    dim = length(initial_point);

    % Khởi tạo biến symbolic
    x = sym('x', [1, dim]);  % Khai báo một vector cột gồm 'dim' biến symbolic

    % Tính gradient của hàm mục tiêu
    gradf = gradient(f(x), x);

    % Chuyển hàm symbolic gradient thành hàm số
    grad_f = matlabFunction(gradf, 'Vars', {x});

    x = initial_point; % Ensure x is a row vector
    v = zeros(size(x)); % Initialize velocity vector
    path = x;
    iter = 0;
    while iter < max_iter
        grad = grad_f(x);
        v = beta * v - alpha * grad';
        x_new = x + v;
        path = [path; x_new];
        if norm(x_new - x, 2) < tol
            break;
        end
        x = x_new;
        iter = iter + 1;
    end
    elapsedTime = toc; % end timing
    xmin = x;
    fmin = f(x);
    
    if dim ==2 
        figure;
        % Create a grid of points for the contour plot
        [X, Y] = meshgrid(linspace(-5, 5, 100), linspace(-5, 5, 100));
        Z = arrayfun(@(x, y) f([x, y]), X, Y);
    
        % Plot the contour
        contour(X, Y, Z, 50); % 50 contour levels
        hold on;
    
        % Plot the best solutions
        plot(path(:, 1), path( :, 2), 'r-o', 'LineWidth', 2, 'MarkerFaceColor', 'r');
    
        % Plot the last solution with a different color
        plot(target_point(1), target_point(2), 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
        
        xlabel('x');
        ylabel('y');
        title('Contour Plot with Best Solution Path');
        grid on;
        hold off;
    end
end

