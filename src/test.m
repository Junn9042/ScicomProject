% Define the function to minimize
rosenbrock = @(x)(1 - x(1))^2 + 100*(x(2) - x(1)^2)^2;

goldstein_price = @(x)(1 + (x(1) + x(2) + 1)^2 * (19 - 14*x(1) + 3*x(1)^2 - 14*x(2) + 6*x(1)*x(2) + 3*x(2)^2)) * (30 + (2*x(1) - 3*x(2))^2 * (18 - 32*x(1) + 12*x(1)^2 + 48*x(2) - 36*x(1)*x(2) + 27*x(2)^2));

f = @(x) x^4 - 4*x^3 + x^2 + 9*x;


max_iterations = 1000;
kMax = 5;
initial_point = [1.43, -1.54];
target_point = [0, -1];
alpha = 0.01;
beta = 0.4;
tol = 1e-5;
epsilon = 0.5;

[bestSol, bestCost, elapsed_time] = quasi_newton_method(goldstein_price, initial_point, tol, max_iter, target_point);

% Display results
disp('Best Solution:');
disp(bestSol);
disp('Best Cost:');
disp(bestCost);
disp('Elapsed Time:');
disp(elapsed_time);