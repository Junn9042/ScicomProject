% Khởi tạo giá trị ban đầu cho x1 và x2
x1_init = 1.1;
x2_init = 2.2;
x = [x1_init, x2_init];
% xstart = x; % Lưu giá trị ban đầu để tính khoảng cách
% 
f =  @(x)(1 + (x(1) + x(2) + 1)^2 * (19 - 14*x(1) + 3*x(1)^2 - 14*x(2) + 6*x(1)*x(2) + 3*x(2)^2)) * (30 + (2*x(1) - 3*x(2))^2 * (18 - 32*x(1) + 12*x(1)^2 + 48*x(2) - 36*x(1)*x(2) + 27*x(2)^2));


% gf = @(x) [100*2*(x(2)-x(1)^2)*(-2*x(1)) + 2*x(1)-2; 200*(x(2)-x(1)^2)];
% hf = @(x) [1200*x(1)^2 - 400*x(2)+ 2, -400*x(1); -400*x(1), 200];
% 
% fval = f(x); gval = gf(x); H = hf(x); ng = norm(gval); nf = 1; tol = 0.05; iter = 0;

tol = 0.05;
max_iter = 100;
target_point = [0, -1];
[x_final, fval_final, iter_final, x1_history, x2_history] = newton_method(x, f, tol, max_iter, target_point);

function [x, fval, iter, x1_history, x2_history] = newton_method(x0, f, tol, max_iter, target_point)

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
    fval = f(x);
    syms x1 x2;
    
    f_sym = f([x1, x2]);
    grad_f = gradient(f_sym, [x1, x2]);
    gval = double(subs(grad_f, [x1, x2], [x(1), x(2)]));
    
    H_matrix = hessian(f_sym, [x1, x2]);
    H = double(subs(H_matrix, [x1, x2], [x(1), x(2)]));
    
    ng = norm(gval); tol = 0.05; iter = 0;

    % Lưu trữ lịch sử giá trị của x1 và x2 để vẽ đồ thị
    x1_history = x(1);
    x2_history = x(2);

    while ng >= tol && iter < max_iter % Giới hạn số vòng lặp tối đa là 50
        iter = iter + 1;
        nf = 0;
        
%         p: Hướng di chuyển trong mỗi bước lặp.
        p = -inv(H)*gval;
       
        % alpha: Bước đi trong quá trình tìm kiếm đường dẫn.
        alpha = 1;
        
%         c1, c2: Hằng số để kiểm tra điều kiện trong quá trình tìm kiếm đường dẫn.
        c1 = 1e-4; c2 = 0.9;
        
%         phi_0, dphi_0: Giá trị của hàm mục tiêu và đạo hàm tại điểm x.
        phi_0 = fval;
        dphi_0 = gval'*p;
        
%         alpha_prev: Giá trị của alpha trong vòng lặp trước đó.
        alpha_prev = 0;

        while true
%             phi_alpha, phi_prime_alpha: Giá trị của hàm mục tiêu và đạo hàm tại điểm x + alpha*p.
            phi_alpha = f(x + alpha*p');
            x_new = x + alpha*p';
            phi_prime_alpha = double(subs(grad_f, [x1, x2], [x_new(1), x_new(2)]));

            if phi_alpha > phi_0 + c1*alpha*dphi_0 || (phi_alpha >= phi_0 && nf > 0)
                alpha_star = zoom(f, x, p, alpha_prev, alpha, phi_0, c1, c2, dphi_0);
                break;
            end

            if abs(phi_prime_alpha) <= -c2*dphi_0
%                 alpha_star: Giá trị của alpha tối ưu sau quá trình tìm kiếm đường dẫn.
                alpha_star = alpha;
                break;
            end

            if phi_prime_alpha >= 0
                
%                 alpha_star: Giá trị của alpha tối ưu sau quá trình tìm kiếm đường dẫn.
                alpha_star = zoom(f, x, p, alpha, alpha_prev, phi_0, c1, c2, dphi_0);
                break;
            end

            alpha_prev = alpha;
            alpha = alpha * 2;
            nf = nf + 1;
        end

        x = x + alpha_star*p';
        fval = f(x);
        gval = double(subs(grad_f, [x1, x2], [x(1), x(2)]));
        H = double(subs(H_matrix, [x1, x2], [x(1), x(2)]));
        ng = norm(gval);

        % Lưu lịch sử giá trị của x1 và x2
        x1_history = [x1_history, x(1)];
        x2_history = [x2_history, x(2)];

        fprintf('%3i %3.2e %3.2e %3.2e %3.2e %i\n',iter,fval,ng,norm(x-xstart),alpha_star,nf);
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
    legend('Contours', 'Convergence path', 'Final point');
    hold off;
end

% find alpha
function alpha_star = zoom(f, x, p, lo, hi, phi_0, c1, c2, dphi_0)
    % Tham số
% f: Hàm mục tiêu.
% gf: Hàm gradient của hàm mục tiêu.
% x: Điểm hiện tại.
% p: Hướng di chuyển.
% lo, hi: Khoảng giá trị của alpha cần tìm kiếm.
% phi_0: Giá trị của hàm mục tiêu tại điểm x.
% c1, c2: Hằng số để kiểm tra điều kiện trong quá trình tìm kiếm đường dẫn.
% dphi_0: Giá trị của đạo hàm tại điểm x.
    % ----
    
    
    % Return
    % alpha_star: Giá trị của alpha tối ưu sau quá trình tìm kiếm.
    % ---

    
% phi_lo, dphi_lo: Giá trị của hàm mục tiêu và đạo hàm tại điểm x + lo*p.
% phi_hi, dphi_hi: Giá trị của hàm mục tiêu và đạo hàm tại điểm x + hi*p.
% alpha_j: Giá trị trung bình của lo và hi trong mỗi vòng lặp.
% phi_j, dphi_j: Giá trị của hàm mục tiêu và đạo hàm tại điểm x + alpha_j*p.
    
    syms x1 x2;
    
    f_sym = f([x1, x2]);
    grad_f = gradient(f_sym, [x1, x2]);
    
    phi_lo = f(x + lo*p);
    x_lo = x + lo*p';
    dphi_lo = double(subs(grad_f, [x1, x2], [x_lo(1), x_lo(2)]))'*p;
    phi_hi = f(x + hi*p);
    x_hi = x + hi*p';
    dphi_hi = double(subs(grad_f, [x1, x2], [x_hi(1), x_hi(2)]))'*p;
    
    
% 1. Tính giá trị của hàm mục tiêu và đạo hàm tại các điểm x + lo*p (gán cho phi_lo và dphi_lo) 
% và x + hi*p (gán cho phi_hi và dphi_hi).

% 2. Thực hiện vòng lặp while cho đến khi tìm được giá trị alpha_star tối ưu:
% 
% Kiểm tra điều kiện dừng dựa trên giá trị của phi_lo, phi_hi và các hằng số c1, c2:
% 
%   Nếu phi_lo không thỏa mãn điều kiện, trả về alpha_star = hi.
%   Nếu phi_hi không thỏa mãn điều kiện, trả về alpha_star = lo.
% 
% 
% Nếu cả phi_lo và phi_hi đều không thỏa mãn điều kiện, tính giá trị alpha_j là điểm giữa của lo và hi, 
% và tính giá trị phi_j và dphi_j tại điểm x + alpha_j*p.

% Kiểm tra điều kiện để cập nhật khoảng giá trị lo và hi dựa trên phi_j và dphi_j:
% 
%   Nếu phi_j không thỏa mãn điều kiện, thu hẹp khoảng tìm kiếm bằng cách 
%   cập nhật hi = alpha_j và cập nhật phi_hi và dphi_hi.

% Nếu phi_j thỏa mãn điều kiện:
% 
%       Nếu abs(dphi_j) <= -c2*dphi_0, trả về alpha_star = alpha_j.
%       Nếu dphi_j và (hi - lo) cùng dấu, thu hẹp khoảng tìm kiếm bằng cách cập nhật hi = lo.

% Cập nhật lo = alpha_j, phi_lo = phi_j và dphi_lo = dphi_j.
    
    while true
        if phi_lo > phi_0 + c1*lo*dphi_0 || (phi_lo >= phi_0 && dphi_lo >= 0)
            alpha_star = hi;
            return;
        elseif phi_hi > phi_0 + c1*hi*dphi_0 || (phi_hi >= phi_0 && dphi_hi >= 0)
            alpha_star = lo;
            return;
        else
            alpha_j = (lo + hi)/2;
            phi_j = f(x + alpha_j*p);
            x_alpha_j = x + alpha_j*p';
            dphi_j = double(subs(grad_f, [x1, x2], [x_alpha_j(1), x_alpha_j(2)]))'*p;
            
            if phi_j > phi_0 + c1*alpha_j*dphi_0 || (phi_j >= phi_lo)
                hi = alpha_j;
                phi_hi = phi_j;
                dphi_hi = dphi_j;
            else
                if abs(dphi_j) <= -c2*dphi_0
                    alpha_star = alpha_j;
                    return;
                end
                if dphi_j*(hi - lo) >= 0
                    hi = lo;
                end
                lo = alpha_j;
                phi_lo = phi_j;
                dphi_lo = dphi_j;
            end
        end
    end
end


