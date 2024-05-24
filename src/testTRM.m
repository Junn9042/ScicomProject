% Ví dụ gọi hàm trust_region_method từ bên ngoài
f = @(x)(1 + (x(1) + x(2) + 1)^2 * (19 - 14*x(1) + 3*x(1)^2 - 14*x(2) + 6*x(1)*x(2) + 3*x(2)^2)) * (30 + (2*x(1) - 3*x(2))^2 * (18 - 32*x(1) + 12*x(1)^2 + 48*x(2) - 36*x(1)*x(2) + 27*x(2)^2));
initial_point = [3, 3];
target_point = [0, -1];
max_iterations = 10000;
[x, fval, elapsed_time] = trust_region_method(f, initial_point, target_point, max_iterations);
disp('Best Solution:');
disp(x);

disp('Best Cost:');
disp(fval);
