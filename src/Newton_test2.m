% Khởi tạo giá trị ban đầu cho x1 và x2
x1_init = 1.1;
x2_init = 2.2;
x = [x1_init, x2_init];
% xstart = x; % Lưu giá trị ban đầu để tính khoảng cách
% 
f = @(x) 100*(x(2)-x(1)^2)^2 + (1-x(1))^2;
% gf = @(x) [100*2*(x(2)-x(1)^2)*(-2*x(1)) + 2*x(1)-2; 200*(x(2)-x(1)^2)];
% hf = @(x) [1200*x(1)^2 - 400*x(2)+ 2, -400*x(1); -400*x(1), 200];
% 
% fval = f(x); gval = gf(x); H = hf(x); ng = norm(gval); nf = 1; tol = 0.05; iter = 0;
target_point = [1, 1];
tol = 1e-6;
max_iter = 1000;
[x_final, fval_final, iter_final, x1_history, x2_history] = newton_method(x, f, max_iter, tol, target_point);

function [x, fval, iter, x1_history, x2_history] = newton_method(x0, f, max_iter, tol, target_point)
    % --- 
    % Tham số
    %     x0: Điểm ban đầu (giá trị khởi tạo) cho vector x (gồm x1 và x2).
    %     f: Hàm mục tiêu (objective function) cần tối ưu.
    %     gf: Hàm gradient (đạo hàm) của hàm mục tiêu f.
    %     f: Hàm ma trận Hessian (ma trận đạo hàm bậc hai) của hàm mục tiêu f.
    % ----
            
    % ---
%     Return
%     x: Giá trị cuối cùng (điểm hội tụ) của vector x sau quá trình lặp.
%     fval: Giá trị của hàm mục tiêu f tại điểm hội tụ x.
%     iter: Số vòng lặp đã thực hiện.
%     x1_history: Mảng lịch sử giá trị của x1 qua các vòng lặp để vẽ đồ thị.
%     x2_history: Mảng lịch sử giá trị của x2 qua các vòng lặp để vẽ đồ thị.
    % ---
    
%     x1_init, x2_init: Giá trị ban đầu của x1 và x2.
    x1_init = x0(1);
    x2_init = x0(2);
    
%   x: Vector chứa giá trị hiện tại của x1 và x2.
    x = [x1_init, x2_init];
%   Lưu giá trị ban đầu để tính khoảng cách
    xstart = x; 
    
%     fval: Giá trị của hàm mục tiêu f tại điểm x.
%     gval: Giá trị của gradient gf tại điểm x.
%     H: Giá trị của ma trận Hessian hf tại điểm x.
%     ng: Chuẩn (norm) của gval, được sử dụng làm điều kiện dừng.
%     nf: Số lần tính toán hàm trong quá trình tìm kiếm đường dẫn (line search).
%     tol: Ngưỡng dung sai (tolerance) để dừng vòng lặp.
%     iter: Số vòng lặp hiện tại.
    
    fval = f(x); gval = numerical_gradient(f, x); H = numerical_hessian(f, x); 
    ng = norm(gval); iter = 0; nf = 1; alpha  = 1;

    % Lưu trữ lịch sử giá trị của x1 và x2 để vẽ đồ thị
    x1_history = x(1);
    x2_history = x(2);

    while ng >= tol && iter < max_iter % Giới hạn số vòng lặp tối đa là 50
        iter = iter + 1;
        
    %   p: Hướng di chuyển trong mỗi bước lặp.
        p = -inv(H)*gval;
        
        % alpha: Bước đi trong quá trình tìm kiếm đường dẫn.
        alpha = 1;
        alpha_min = 1e-6; % Giá trị nhỏ nhất của alpha
        
        nf = 0;
        pass = 0;
        
        while pass == 0
            ftest = f(x+alpha*p');
            nf = nf+1;
        if ftest <= fval + 0.01*alpha*gval'*p
            pass = 1;
            x = x+alpha*p';
        % Lưu lịch sử giá trị của x1 và x2
            x1_history = [x1_history, x(1)];
            x2_history = [x2_history, x(2)];
        
            fval = ftest;
            gval = numerical_gradient(f, x);
            H = numerical_hessian(f, x);
            ng = norm(gval);
        else
            if alpha < alpha_min
                % Không thể tìm thấy alpha hợp lệ, giữ nguyên x và thoát khỏi vòng lặp
                break;
            end
                alpha = alpha / 2; % Giảm alpha nếu không thỏa mãn điều kiện
        end
        end
    end
    
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
    plot(target_point(1), target_point(2), 'bx', 'LineWidth', 2, 'MarkerSize', 10);
    
    xlabel('x1');
    ylabel('x2');
    title('Contour plot and convergence path');
    legend('Contours', 'Convergence path', 'Target point');
    hold off;
end

% Hàm tính gradient
function g = numerical_gradient(f, x0)
    epsilon = 1e-6; % Giá trị epsilon nhỏ
    n = numel(x0);
    g = zeros(n, 1);
    fx = f(x0);
    
    for i = 1:n
        x_temp = x0;
        x_temp(i) = x_temp(i) + epsilon;
        g(i) = (f(x_temp) - fx) / epsilon;
    end
end

function H = numerical_hessian(f, x0)
    epsilon = 1e-4; % Giá trị epsilon nhỏ
    n = numel(x0);
    H = zeros(n, n);
    
    for i = 1:n
        for j = 1:n
            x_ij1 = x0;
            x_ij2 = x0;
            x_ij3 = x0;
            x_ij4 = x0;
            
            x_ij1([i, j]) = x_ij1([i, j]) + epsilon;
            x_ij2(i) = x_ij2(i) + epsilon; x_ij2(j) = x_ij2(j) - epsilon;
            x_ij3(i) = x_ij3(i) - epsilon; x_ij3(j) = x_ij3(j) + epsilon;
            x_ij4([i, j]) = x_ij4([i, j]) - epsilon;
            
            H(i, j) = (f(x_ij1) - f(x_ij2) - f(x_ij3) + f(x_ij4)) / (4 * epsilon^2);
        end
    end
end






