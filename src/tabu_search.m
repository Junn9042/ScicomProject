function [best_solution, best_cost, elapsed_time] = tabu_search(f, initial_point, target_point, max_iterations, neighborhood_size, tabu_list_size)
    % Start timer
    tic;
    
    % Initialization
    current_solution = initial_point;
    best_solution = current_solution;
    best_cost = f(current_solution);

    % Tabu list
    tabu_list = NaN(tabu_list_size, numel(initial_point));

    % Arrays to store the iteration numbers and best costs for plotting
    best_solutions = [initial_point];

    % Tabu Search Algorithm
    for iter = 1:max_iterations
        % Generate neighbors
        neighbors = generate_neighbors(current_solution, neighborhood_size);
        
        % Evaluate neighbors
        neighbor_costs = zeros(size(neighbors, 1), 1);
        for i = 1:size(neighbors, 1)
            neighbor_costs(i) = f(neighbors(i, :));
        end
        
        % Find best non-tabu neighbor
        best_neighbor_idx = -1;
        best_neighbor_cost = inf;
        for i = 1:size(neighbors, 1)
            if neighbor_costs(i) < best_neighbor_cost && ~is_tabu(neighbors(i, :), tabu_list)
                best_neighbor_idx = i;
                best_neighbor_cost = neighbor_costs(i);
            end
        end
        
        % Update current solution and tabu list
        current_solution = neighbors(best_neighbor_idx, :);
        tabu_list = [tabu_list(2:end, :); current_solution]; % Add the new move to tabu list
        
        % Update best solution if needed
        if best_neighbor_cost < best_cost
            best_solution = current_solution;
            best_cost = best_neighbor_cost;

            % Store the iteration number and best cost for plotting
            best_solutions = [best_solutions; best_solution];
        end
    end

    % Stop timer and calculate running time
    elapsed_time = toc;

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
        plot(best_solutions(:, 1), best_solutions(:, 2), 'r-o', 'LineWidth', 2, 'MarkerFaceColor', 'r');
    
        % Plot the last solution with a different color
        plot(target_point(1), target_point(2), 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
        
        xlabel('x');
        ylabel('y');
        title('Contour Plot with Best Solution Path - Tabu Search');
        grid on;
        hold off;
   end
end

% Function to generate neighboring solutions
function neighbors = generate_neighbors(solution, size)
    % Perturb each variable in the solution to generate neighbors
    perturbation = 0.5; % Perturbation factor
    neighbors = repmat(solution, size, 1) + perturbation * randn(size, numel(solution));
end

% Function to check if a move is tabu
function tabu = is_tabu(move, tabu_list)
    % Check if the move is in the tabu list
    tabu = any(all(tabu_list == move, 2));
end