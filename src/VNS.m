function [bestSol, bestCost, elapsedTime] = VNS(f, initialPoint, targetPoint, MaxIt, kMax)

    tic; 
    
    bestSol = initialPoint;  % Sử dụng điểm bắt đầu cung cấp
    nVar = length(initialPoint);
    
    bestCost = f(bestSol);

    iteration_numbers = [];
    best_costs = [];
    best_solutions = [];

    % Lặp VNS
    for it = 1:MaxIt
        for k = 1:kMax
            % Shake
            newSol = Shake(bestSol, k, nVar);
            % Local Search
            newSol = LocalSearch(newSol, f, nVar);
            newCost = f(newSol);
            
            % Cập nhật nếu tìm thấy giải pháp tốt hơn
            if newCost < bestCost
                bestSol = newSol;
                bestCost = newCost;
                k = 1;  % reset k

                % Store the iteration number and best cost for plotting
                iteration_numbers = [iteration_numbers, it];
                best_costs = [best_costs, bestCost];
                best_solutions = [best_solutions; bestSol];
            end
        end
    end
    
    elapsedTime = toc;  % Lấy thời gian kết thúc


    if length(initialPoint) == 2
         % Plot the contour and the best solutions
        figure;
        % Create a grid of points for the contour plot
        [X, Y] = meshgrid(linspace(-2, 2, 100), linspace(-2, 2, 100));
        Z = arrayfun(@(x, y) f([x, y]), X, Y);
    
        % Plot the contour
        contour(X, Y, Z, 50); % 50 contour levels
        hold on;
    
        % Plot the best solutions
        plot(best_solutions(:, 1), best_solutions( :, 2), 'r-o', 'LineWidth', 2, 'MarkerFaceColor', 'r');
    
        % Plot the last solution with a different color
        plot(targetPoint(1), targetPoint(2), 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
        
        xlabel('x');
        ylabel('y');
        title('Contour Plot with Best Solution Path');
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