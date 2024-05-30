function [best_solution, best_cost, elapsed_time] = VNS(f, initial_point, target_point, max_iterations, kMax)

    tic; 
    
    best_solution = initial_point;  % Sử dụng điểm bắt đầu cung cấp
    nVar = length(initial_point);
    
    best_cost = f(best_solution);
    best_solutions = [initial_point];

    % Lặp VNS
    for it = 1:max_iterations
        for k = 1:kMax
            % Shake
            newSol = Shake(best_solution, k, nVar);
            % Local Search
            newSol = LocalSearch(newSol, f, nVar);
            newCost = f(newSol);
            
            % Cập nhật nếu tìm thấy giải pháp tốt hơn
            if newCost < best_cost
                best_solution = newSol;
                best_cost = newCost;
                k = 1;  % reset k

                % Store the iteration number and best cost for plotting
                best_solutions = [best_solutions; best_solution];
            end
        end
    end
    
    elapsed_time = toc;  % Lấy thời gian kết thúc


    if length(initial_point) == 2
         % Plot the contour and the best solutions
        figure;
        % Create a grid of points for the contour plot
        [X, Y] = meshgrid(linspace(min(best_solutions(:, 1)) - 1, max(best_solutions(:, 1)) + 1 , 100), linspace(min(best_solutions(:, 2)) - 1, max(best_solutions(:, 2)) + 1, 100));
        Z = arrayfun(@(x, y) f([x, y]), X, Y);
    
        % Plot the contour
        contour(X, Y, Z, 200); % 50 contour levels
        hold on;
    
        % Plot the best solutions
        plot(best_solutions(:, 1), best_solutions( :, 2), 'r-o', 'LineWidth', 2, 'MarkerFaceColor', 'r');
    
        % Plot the last solution with a different color
        plot(target_point(:, 1), target_point(:, 2), 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
        
        xlabel('x');
        ylabel('y');
        title('Contour Plot with Best Solution Path - Variable Neighbour Search');
        grid on;
        hold off;
    
    end
           
   

    function newSol = Shake(x, k, nVar)
        sigma = 0.2 * k;  % Cường độ rung động dựa trên k
        %disp(size(sigma));
        newSol = x + sigma * randn(1, nVar); % Rung động không giới hạn
        %disp(size(sigma * randn(1, nVar)'));
        %disp(size(x));
    end

    function x = LocalSearch(x, f, nVar)
        for i = 1:50
            localSol = x + 0.01 * randn(1, nVar);
            if f(localSol) < f(x)
                x = localSol;
            end
        end
end
end