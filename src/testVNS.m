% Define the function to minimize
rosenbrock = @(x)(1 - x(1))^2 + 100*(x(2) - x(1)^2)^2;

gp = @(x)(1 + (x(1) + x(2) + 1)^2 * (19 - 14*x(1) + 3*x(1)^2 - 14*x(2) + 6*x(1)*x(2) + 3*x(2)^2)) * (30 + (2*x(1) - 3*x(2))^2 * (18 - 32*x(1) + 12*x(1)^2 + 48*x(2) - 36*x(1)*x(2) + 27*x(2)^2));

f = @(x) x^4 - 4*x^3 + x^2 + 9*x;


MaxIt = 1000;
kMax = 5;
initialPoint = [-5, 5];

% Call VNS
[bestSol, bestCost, elapsedTime] = VNS(gp, initialPoint, [0, -1], MaxIt, kMax);

% Display results
disp('Best Solution:');
disp(bestSol);
disp('Best Cost:');
disp(bestCost);
disp('Elapsed Time:');
disp(elapsedTime);