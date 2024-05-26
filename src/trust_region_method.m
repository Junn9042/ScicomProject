function [x, fval, elapsed_time] = trust_region_method(f, initial_point, target_point, max_iterations)
    % Initialize B as identity matrix
    B = eye(length(initial_point));
    
    % Start counting time
    tic;
    
     % Khởi tạo
    x_sym = sym('x', [length(initial_point), 1]); % Biến symbolic
    x = initial_point(:); % Chuyển x0 thành vector cột

    Delta = 1;
    tolFun = 1e-6;
    tolX = 1e-6;
    eta1 = 0.75;
    eta2 = 0.375;
    iter = 0;

    best_solutions = [initial_point'];
    dim = length(initial_point);

    % Tính gradient của hàm mục tiêu
    gradf = gradient(f(x_sym), x_sym);

    % Chuyển hàm symbolic gradient thành hàm số
    gradFunc = matlabFunction(gradf, 'Vars', {x_sym});
    
    while iter < max_iterations
        iter = iter + 1;
        
        % Compute function value and gradient
        fval = f(x);
        grad = gradFunc(x);

        % Build quadratic model
        m = @(p) fval + grad' * p + 0.5 * p' * B * p;
        
        % Solve trust region subproblem
        [p, ~] = dogleg_method(grad, B, Delta);
        
        % Compute new function value at trial point
        x_new = x + p;
        fval_new = f(x_new);
        
        % Compute actual and predicted reduction
        actual_reduction = fval - fval_new;
        predicted_reduction = m(zeros(size(x))) - m(p);
        
        % Compute rho
        rho = actual_reduction / predicted_reduction;
        
        % Update trust region radius
        if rho >= eta1
            x = x_new;
            best_solutions = [best_solutions, x];
            Delta = min(2 * Delta, 1.0); % Giới hạn tăng Delta
        elseif rho > eta2
            x = x_new;
            best_solutions = [best_solutions, x];
        else
            Delta = Delta / 2;
        end
        
        % Check convergence
        if abs(fval - fval_new) < tolFun || norm(p) < tolX
            break;
        end
        
        % Update B using BFGS
        s = p;
        y = gradient(f(x_new)) - grad;
        B = update_BFGS(B, s, y);
    end
    
    % End counting time
    elapsed_time = toc;
    if length(initial_point) == 2
        
        % Create a grid of points for the contour plot
        [X, Y] = meshgrid(linspace(-5, 5, 100), linspace(-5, 5, 100));
        Z = arrayfun(@(x, y) f([x, y]), X, Y);
        
        % Plot the contour
        figure;
        contour(X, Y, Z, 200); % 50 contour levels
        hold on;
        
        % Plot the best solutions
        plot(best_solutions(1, :), best_solutions(2, :), 'r-o', 'LineWidth', 2, 'MarkerFaceColor', 'r');
        
        % Plot the last solution with a different color
        plot(target_point(1), target_point(2), 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
        
        xlabel('x');
        ylabel('y');
        title('Contour Plot with Best Solution Path');
        grid on;
        hold off;
    end
end

function B = update_BFGS(B, s, y)
    % Update B using BFGS formula
    rho = 1 / (y' * s);
    term1 = (eye(length(B)) - rho * s * y');
    term2 = (eye(length(B)) - rho * y * s');
    B = term1 * B * term2 + rho * s * s';
end

function [p, fval] = dogleg_method(grad, B, Delta)
    % Solve trust region subproblem using dogleg method
    pU = -Delta * grad / norm(grad); % Compute Cauchy point
    if norm(B * grad) <= Delta
        pB = -B \ grad; % Full step (Newton step)
    else
        % Compute scaled Newton step
        pB = -(grad' * grad) / (grad' * B * grad) * grad;
    end
    if norm(pB) <= Delta
        p = pB; % Step lies inside the trust region
    else
        % Compute dogleg step
        pD = find_dogleg_step(grad, B, Delta);
        if norm(pD) >= norm(pU)
            p = pU; % Choose Cauchy step
        else
            p = pD; % Choose dogleg step
        end
    end
    fval = 0.5 * p' * B * p + grad' * p;
end

function p = find_dogleg_step(grad, B, Delta)
    % Compute dogleg step using polynomial interpolation
    pGN = -B \ grad; % Gauss-Newton step (Newton step)
    if norm(pGN) <= Delta
        p = pGN; % Step lies inside the trust region
    else
        pSD = -grad * (grad' * grad) / (grad' * B * grad); % Steepest descent step
        if norm(pSD) >= Delta
            p = Delta * pSD / norm(pSD); % Step lies on the boundary
        else
            % Compute coefficients of quadratic interpolation
            a = pGN' * pGN;
            b = 2 * pGN' * (pSD - pGN);
            c = (pSD - pGN)' * (pSD - pGN) - Delta^2;
            tau = (-b + sqrt(b^2 - 4 * a * c)) / (2 * a);
            p = pGN + tau * (pSD - pGN); % Dogleg step
        end
    end
end