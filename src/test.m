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

%Branin RCOS
%global (pi, 2.275)
%search domain (-5, 5)x(-5, 5)
branin = @(x)1 * (x(2) - 5.1 / (4 * pi^2) * x(1)^2 + 5 / pi * x(1) - 6)^2 + 10 * (1 - 1 / (8 * pi)) * cos(x(1)) + 10;

%Hartman3-4
% Constants
alpha = [1.0, 1.2, 3.0, 3.2];
A = [3.0, 10.0, 30.0;
     0.1, 10.0, 35.0;
     3.0, 10.0, 30.0;
     0.1, 10.0, 35.0];
P = [0.3689, 0.1170, 0.2673;
     0.4699, 0.4387, 0.7470;
     0.1091, 0.8732, 0.5547;
     0.0381, 0.5743, 0.8828];

% Hartman 3-dimensional function as an anonymous function
% Global (0.11, 0.555, 0.855)
% 4 local minima
%Search domain 0 < xj < 1; j = 1, 3 
hartman3 = @(x) -(...
    alpha(1) * exp(- (A(1, 1) * (x(1) - P(1, 1))^2 + A(1, 2) * (x(2) - P(1, 2))^2 + A(1, 3) * (x(3) - P(1, 3))^2)) + ...
    alpha(2) * exp(- (A(2, 1) * (x(1) - P(2, 1))^2 + A(2, 2) * (x(2) - P(2, 2))^2 + A(2, 3) * (x(3) - P(2, 3))^2)) + ...
    alpha(3) * exp(- (A(3, 1) * (x(1) - P(3, 1))^2 + A(3, 2) * (x(2) - P(3, 2))^2 + A(3, 3) * (x(3) - P(3, 3))^2)) + ...
    alpha(4) * exp(- (A(4, 1) * (x(1) - P(4, 1))^2 + A(4, 2) * (x(2) - P(4, 2))^2 + A(4, 3) * (x(3) - P(4, 3))^2)) ...
    );

%Beale function
%search domain -2.5 < x1 < 4.5 , -2 < x2 < 2
%global minimum : (3, 0.5)
beale = @(x) (1.5 - x(1) + x(1) * x(2))^2 + (2.25 - x(1) + x(1) * x(2)^2)^2 + (2.625 - x(1) + x(1) * x(2)^3)^2;

% Booth function 
%search domain -10 < xi < 10
%global (1, 3)
booth = @(x) (x(1) + 2*x(2) - 7)^2 + (2*x(1) + x(2) - 5)^2;


max_iterations = 1000;
kMax = 5;
initial_point = [0.7, 0.5, 1];
target_point = [0.11, 0.555, 0.855];
alpha = 0.01;
beta = 0.4;
tol = 1e-6;
epsilon = 0.5;
neighborhood_size = 10;
tabu_list_size = 5;

test_all(hartman3, initial_point, target_point);

function [] = test_all(f, initial_point, target_point)
    max_iterations = 1000;
    TS_max_iterations = 5000;
    kMax = 5;
    alpha = 0.01;
    beta = 0.4;
    tol = 1e-5;
    epsilon = 0.5;
    neighborhood_size = 10;
    tabu_list_size = 5;
    [best_sol_GD, best_cost_GD, elapsed_time_GD] = gradient_descent(f, initial_point, target_point, alpha, max_iterations, tol, epsilon);
    [best_sol_N, best_cost_N, elapsed_time_N] = newton_method(f, initial_point,target_point, max_iterations, tol);
    [best_sol_QN, best_cost_QN, elapsed_time_QN] = quasi_newton_method(f, initial_point, target_point, tol, max_iterations);
    [best_sol_TR, best_cost_TR, elapsed_time_TR] = trust_region_method(f, initial_point, target_point, max_iterations);
    [best_sol_TS, best_cost_TS, elapsed_time_TS] = tabu_search(f, initial_point, target_point, TS_max_iterations, neighborhood_size, tabu_list_size);
    [best_sol_VNS, best_cost_VNS, elapsed_time_VNS] =  VNS(f, initial_point, target_point, max_iterations, kMax);
    [best_sol_GDm, best_cost_GDm, elapsed_time_GDm] = gradient_descent_with_momentum(f, initial_point, target_point, alpha, beta, max_iterations, tol);

    fprintf ("%-30s%-30s%-20s%-20s\n", "Method", "Minimum point", "Minimum value", "Elapsed time");
    fprintf ("%-30s%-30s%-20.3f%-20.3f\n", "GD with changed alpha", strjoin(compose('%.3f', best_sol_GD), ' '), best_cost_GD, elapsed_time_GD);
    fprintf ("%-30s%-30s%-20.3f%-20.3f\n", "GD with momentum",strjoin(compose('%.3f', best_sol_GDm), ' '), best_cost_GDm, elapsed_time_GDm);
    fprintf ("%-30s%-30s%-20.3f%-20.3f\n", "Newton", strjoin(compose('%.3f', best_sol_N), ' '), best_cost_N, elapsed_time_N);
    fprintf ("%-30s%-30s%-20.3f%-20.3f\n", "Quasi Newton", strjoin(compose('%.3f', best_sol_QN), ' '), best_cost_QN, elapsed_time_QN);
    fprintf ("%-30s%-30s%-20.3f%-20.3f\n", "Trust Region", strjoin(compose('%.3f', best_sol_TR), ' '), best_cost_TR, elapsed_time_TR);
    fprintf ("%-30s%-30s%-20.3f%-20.3f\n", "Tabu Search", strjoin(compose('%.3f', best_sol_TS), ' '), best_cost_TS, elapsed_time_TS);
    fprintf ("%-30s%-30s%-20.3f%-20.3f\n", "VNS", strjoin(compose('%.3f', best_sol_VNS), ' '), best_cost_VNS, elapsed_time_VNS);
end
