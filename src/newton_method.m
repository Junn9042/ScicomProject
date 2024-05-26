function [best_solution, best_cost, elapsed_time] = newton_method(f, initial_point,target_point, max_iterations, tol)
    % This function uses Newton's method to solve an unconstrained minimization problem and plots the path of the solution.
    % Inputs:
    %   f              - Function handle of the objective function
    %   initial_point  - Initial guess for the minimizer
    %   max_iterations - Maximum number of iterations
    %   tol            - Tolerance for the stopping criterion
    % Outputs:
    %   best_solution  - The estimated solution
    %   best_cost      - The value of the objective function at the best solution
    %   elapsed_time   - The time taken to find the solution

    % Start timer
    tic;
    
    % Convert the function handle to a symbolic function
    initial_point = initial_point(:); % Ensure it's a column vector
    syms x [length(initial_point) 1]
    symbolic_f = f(x);
    
    % Calculate the gradient (first derivative) and Hessian (second derivative)
    grad_f = gradient(symbolic_f, x);
    hessian_f = hessian(symbolic_f, x);
    
    % Convert symbolic expressions to function handles for numerical evaluation
    grad_f = matlabFunction(grad_f, 'Vars', {x});
    hessian_f = matlabFunction(hessian_f, 'Vars', {x});
    
    % Initialize variables
    current_point = initial_point;
    path = current_point'; % Record the initial point
    
    for iter = 1:max_iterations
        % Evaluate the gradient and Hessian at the current point
        grad_value = grad_f(current_point);
        hessian_value = hessian_f(current_point);
        
        % Check if the gradient norm is below the tolerance
        if norm(grad_value) < tol
            break;
        end
        
        % Compute the Newton step
        step = -hessian_value \ grad_value;
        
        % Update the current point
        current_point = current_point + step;
        
        % Record the current point in the path
        path = [path; current_point'];
    end
    
    % Calculate the final cost
    best_cost = double(subs(symbolic_f, x, current_point));
    
    % Set the best solution
    best_solution = current_point';
    
    % Stop timer
    elapsed_time = toc;
    
    if length(initial_point) == 2
        
        % Create a grid of points for the contour plot
        [X, Y] = meshgrid(linspace(-5, 5, 100), linspace(-5, 5, 100));
        Z = arrayfun(@(x, y) f([x, y]), X, Y);
        
        % Plot the contour
        figure;
        contour(X, Y, Z, 50); % 50 contour levels
        hold on;
        
        % Plot the best solutions
        plot(path(:, 1), path(:, 2), 'r-o', 'LineWidth', 2, 'MarkerFaceColor', 'r');
        
        % Plot the last solution with a different color
        plot(target_point(1), target_point(2), 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
        
        xlabel('x');
        ylabel('y');
        title('Contour Plot with Best Solution Path');
        grid on;
        hold off;
    end
end