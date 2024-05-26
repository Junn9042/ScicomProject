% Define the function to minimize

% Rosenbrock: 
% global (1,1)
% search domain (-5,10)x(-5,10)
rosenbrock = @(x)(1 - x(1))^2 + 100*(x(2) - x(1)^2)^2; 

% Goldstein and Price:
% global (0, -1); local (+-0.6, +-0.4)
% search domain (-2,2)x(-2,2)
goldstein_price = @(x)(1 + (x(1) + x(2) + 1)^2 * (19 - 14*x(1) + 3*x(1)^2 - 14*x(2) + 6*x(1)*x(2) + 3*x(2)^2)) * (30 + (2*x(1) - 3*x(2))^2 * (18 - 32*x(1) + 12*x(1)^2 + 48*x(2) - 36*x(1)*x(2) + 27*x(2)^2));

% Quadratic function: 
% global -0.7139; local 2.4018
quadratic = @(x) x^4 - 4*x^3 + x^2 + 9*x;

% ES: 
% global (pi, pi) 
% search domain (-100,100)x(-100,100)
es = @(x)-cos(x(1))* cos(x(2))*exp(-((x(1) - pi)^2 + (x(2) - pi)^2));

max_iterations = 1000;
kMax = 5;
initial_point = [-3, -3];
target_point = [0, -1];
alpha = 0.01;
beta = 0.4;
tol = 1e-5;
epsilon = 0.5;

[bestSol, bestCost, elapsed_time] =  trust_region_method(goldstein_price, initial_point, target_point, max_iterations);
% Display results
disp('Best Solution:');
disp(bestSol);
disp('Best Cost:');
disp(bestCost);
disp('Elapsed Time:');
disp(elapsed_time);