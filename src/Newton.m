% Khởi tạo giá trị ban đầu cho x1 và x2
% x1_init = 1.1;
% x2_init = 2.2;
% x = [x1_init, x2_init];
% xstart = x; % Lưu giá trị ban đầu để tính khoảng cách
% 
% f = @(x) 100*(x(2)-x(1)^2)^2 + (1-x(1))^2;
% gf = @(x) [100*2*(x(2)-x(1)^2)*(-2*x(1)) + 2*x(1)-2; 200*(x(2)-x(1)^2)];
% hf = @(x) [1200*x(1)^2 - 400*x(2)+ 2, -400*x(1); -400*x(1), 200];
% 
% fval = f(x); gval = gf(x); H = hf(x); ng = norm(gval); nf = 1; tol = 0.05; iter = 0;
% [x_final, fval_final, iter_final, x1_history, x2_history] = newton_method(x, f, gf, hf);

function [x, fval, iter, x1_history, x2_history] = newton_method(x0, f, gf, hf)
    % Khởi tạo giá trị ban đầu cho x1 và x2
    x1_init = x0(1);
    x2_init = x0(2);
    x = [x1_init, x2_init];
    xstart = x; % Lưu giá trị ban đầu để tính khoảng cách
    fval = f(x); gval = gf(x); H = hf(x); ng = norm(gval); nf = 1; tol = 0.05; iter = 0;

    % Lưu trữ lịch sử giá trị của x1 và x2 để vẽ đồ thị
    x1_history = x(1);
    x2_history = x(2);

    while ng >= tol && iter < 50 % Giới hạn số vòng lặp tối đa là 50
        iter = iter + 1;
        nf = 0;
        p = -inv(H)*gval;

        % Line search để tìm alpha
        alpha = 1;
        c1 = 1e-4; c2 = 0.9;
        phi_0 = fval;
        dphi_0 = gval'*p;
        alpha_prev = 0;

        while true
            phi_alpha = f(x + alpha*p);
            phi_prime_alpha = gf(x + alpha*p)'*p;

            if phi_alpha > phi_0 + c1*alpha*dphi_0 || (phi_alpha >= phi_0 && nf > 0)
                alpha_star = zoom(f, gf, x, p, alpha_prev, alpha, phi_0, c1, c2, dphi_0);
                break;
            end

            if abs(phi_prime_alpha) <= -c2*dphi_0
                alpha_star = alpha;
                break;
            end

            if phi_prime_alpha >= 0
                alpha_star = zoom(f, gf, x, p, alpha, alpha_prev, phi_0, c1, c2, dphi_0);
                break;
            end

            alpha_prev = alpha;
            alpha = alpha * 2;
            nf = nf + 1;
        end

        x = x + alpha_star*p;
        fval = f(x);
        gval = gf(x);
        H = hf(x);
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
    plot(x(1), x(2), 'bx', 'LineWidth', 2, 'MarkerSize', 10);
    
    xlabel('x1');
    ylabel('x2');
    title('Contour plot and convergence path');
    legend('Contours', 'Convergence path', 'Final point');
    hold off;
end

% find alpha
function alpha_star = zoom(f, gf, x, p, lo, hi, phi_0, c1, c2, dphi_0)
    phi_lo = f(x + lo*p);
    dphi_lo = gf(x + lo*p)'*p;
    phi_hi = f(x + hi*p);
    dphi_hi = gf(x + hi*p)'*p;
    
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
            dphi_j = gf(x + alpha_j*p)'*p;
            
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


