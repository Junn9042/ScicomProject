% Định nghĩa hàm f(x)
f = @(x)(1 + (x(1) + x(2) + 1)^2 * (19 - 14*x(1) + 3*x(1)^2 - 14*x(2) + 6*x(1)*x(2) + 3*x(2)^2)) * (30 + (2*x(1) - 3*x(2))^2 * (18 - 32*x(1) + 12*x(1)^2 + 48*x(2) - 36*x(1)*x(2) + 27*x(2)^2));

% Điểm khởi tạo
x0 = [1; 2];
x1_history = x0(1);
x2_history = x0(2);
% 
% Số lần lặp tối đa và độ chính xác mong muốn
max_iter = 1000;
tol = 1e-6;
% 
% Gọi hàm quasiNewton để tìm điểm cực tiểu và lưu các điểm hội tụ
[x_min, f_min, convergence_points, x1_history, x2_history] = quasiNewton(f, x0, max_iter, tol);
% 
% Vẽ đường đồng mức của hàm
[X,Y] = meshgrid(-2:0.1:2, -1:0.1:3);
Z = 100 * (Y - X.^2).^2 + (1 - X).^2;
contour(X, Y, Z, 20);
hold on;
plot(x1_history, x2_history, 'r-o', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('x1');
ylabel('x2');
title('Contour plot and convergence path');
legend('Contours', 'Convergence path');
hold off;

function [x_min, f_min, convergence_points, x1_history, x2_history] = quasiNewton(f, x0, max_iter, tol)
    % Khởi tạo
    x_k = x0;
    n = length(x0);
    B = eye(n); % Ma trận ước lượng của Hessian ban đầu (là ma trận đơn vị)
    
    x1_history = x0(1);
    x2_history = x0(2);

    % Lặp lại cho đến khi hội tụ
    convergence_points = {};  % Lưu trữ các điểm hội tụ
    for iter = 1:max_iter
        % Bước 3: Tính toán gradient
        gradient = gradientFunction(f, x_k);
        
        % Kiểm tra điều kiện dừng
        if norm(gradient) < tol
            break;
        end
        
        % Bước 4: Tính toán hướng di chuyển
        direction = -B * gradient;
        
        % Bước 5: Tìm kích thước bước bằng phương pháp Armijo
        alpha = armijoLineSearch(f, x_k, direction, gradient);
        
        % Bước 6: Cập nhật vị trí mới
        x_k1 = x_k + alpha * direction;
        
        % Lưu trữ vị trí mới vào cell array
        convergence_points{end+1} = x_k1;
        
        % Bước 7: Tính toán gradient mới tại vị trí mới
        gradient_k1 = gradientFunction(f, x_k1);
        
        % Bước 8: Tính toán sự thay đổi của gradient và vị trí
        s = x_k1 - x_k;
        y = gradient_k1 - gradient;
        
        % Bước 9: Cập nhật ma trận ước lượng của Hessian (BFGS)
        B = B + ((y * y') / (y' * s)) - ((B * s * s' * B) / (s' * B * s));
        
        % Cập nhật vị trí mới
        x_k = x_k1;
        x1_history = [x1_history, x_k(1)];
        x2_history = [x2_history, x_k(2)];
    end

    x_min = x_k;
    f_min = f(x_min);
    
    % Vẽ đồ thị sự hội tụ và các đường đồng mức
    figure;
    [x1_grid, x2_grid] = meshgrid(-2:0.1:2, -1:0.1:3);
    [X1, X2] = deal(x1_grid, x2_grid); % Sử dụng deal để tách X1 và X2
    % Tính toán f_grid bằng cách lặp qua từng phần tử của X1 và X2
    f_grid = zeros(size(X1));
    for i = 1:numel(X1)
        f_grid(i) = f([X1(i), X2(i)]);
    end

    % Kiểm tra kích thước của f_grid và X1
    size_f_grid = numel(f_grid);
    size_X1 = numel(X1);

    if size_f_grid == size_X1
        f_grid = reshape(f_grid, size(X1));
    else
        error('Kích thước của f_grid và X1 không khớp.');
    end
    contour(x1_grid, x2_grid, f_grid, 50); hold on;
    plot(x1_history, x2_history, 'r-o', 'LineWidth', 2, 'MarkerSize', 6);
        
    % Vẽ điểm cuối cùng với màu khác
    plot(x_min(1), x_min(2), 'bx', 'LineWidth', 2, 'MarkerSize', 10);
    
    xlabel('x1');
    ylabel('x2');
    title('Contour plot and convergence path');
    legend('Contours', 'Convergence path', 'Final point');
    hold off;
end

% Hàm tính gradient
function grad = gradientFunction(f, x)
    h = 1e-6;
    n = length(x);
    grad = zeros(size(x));
    for i = 1:n
        x_plus_h = x;
        x_plus_h(i) = x_plus_h(i) + h;
        grad(i) = (f(x_plus_h) - f(x)) / h;
    end
end

% Phương pháp tìm kiếm dòng bằng phương pháp Armijo
function alpha = armijoLineSearch(f, x_k, direction, gradient)
    alpha = 1;
    beta = 0.5;
    c = 0.1;
    while f(x_k + alpha * direction) > f(x_k) + c * alpha * gradient' * direction
        alpha = beta * alpha;
    end
end
