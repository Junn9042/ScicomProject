function [x, fval, elapsed_time] = quasi_newton_method(f, initial_point, tol, max_iterations, target_point)
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

    tic;
    x_sym = sym('x', [length(initial_point), 1]); % Biến symbolic

    % Tính gradient của hàm mục tiêu
    gradf = gradient(f(x_sym), x_sym);

    % Chuyển hàm symbolic gradient thành hàm số
    gradf = matlabFunction(gradf, 'Vars', {x_sym});
    x = initial_point';
    n = length(x);

    H = eye(n);  % Initial approximation to the Hessian is the identity matrix
    iter = 0;
    x_history = initial_point';

    for k = 1:max_iterations
        g = gradf(x);  % Gradient at current point
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
        y = gradf(x_new) - g;  % Change in gradient

        % BFGS update
        H = update_BFGS(H, s, y);
        
        x = x_new;  % Move to the new point
        iter = iter + 1;
        x_history = [x_history, x];
        
    end

    fval = f(x);  % Function value at the found minimum point
    elapsed_time = toc;
    if length(initial_point) == 2
         % Plot the contour and the best solutions
        figure;
        % Create a grid of points for the contour plot
        [X, Y] = meshgrid(linspace(-5, 5, 100), linspace(-5, 5, 100));
        Z = arrayfun(@(x, y) f([x, y]), X, Y);
    
        % Plot the contour
        contour(X, Y, Z, 50); % 50 contour levels
        hold on;
    
        % Plot the best solutions
        plot(x_history(1, :), x_history(2, :), 'r-o', 'LineWidth', 2, 'MarkerFaceColor', 'r');
    
        % Plot the last solution with a different color
        plot(target_point(1), target_point(2), 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
        
        xlabel('x');
        ylabel('y');
        title('Contour Plot with Best Solution Path');
        grid on;
        hold off;
    
    end
end

function H = update_BFGS(H, s, y)
    % Update H using BFGS formula
    rho = 1 / (y' * s);
    term1 = (eye(length(H)) - rho * s * y');
    term2 = (eye(length(H)) - rho * y * s');
    H = term1 * H * term2 + rho * s * s';
end

