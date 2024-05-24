function [x, fval, elapsed_time] = trust_region_method(f, initial_point, target_point, max_iterations)
    % Thiết lập các thông số mặc định nếu không được truyền vào

    % Bắt đầu đếm thời gian
    tic;

    % Khởi tạo
    x_sym = sym('x', [length(initial_point), 1]); % Biến symbolic
    x = initial_point(:); % Chuyển x0 thành vector cột

    Delta = 1;
    tolFun = 1e-6;
    tolX = 1e-6;
    eta1 = 0.5;
    eta2 = 0.2;
    iter = 0;
    best_solutions = [initial_point'];
    dim = length(initial_point);

    % Tính gradient của hàm mục tiêu
    gradf = gradient(f(x_sym), x_sym);
    hessf = hessian(f(x_sym), x_sym);

    % Chuyển hàm symbolic gradient thành hàm số
    gradFunc = matlabFunction(gradf, 'Vars', {x_sym});
    hessFunc = matlabFunction(hessf, 'Vars', {x_sym});

    % Lưu trữ lịch sử x và giá trị hàm mục tiêu
    history = zeros(max_iterations, dim);
    fval_history = zeros(max_iterations, 1);

    while iter < max_iterations
        iter = iter + 1;

        % Tính toán giá trị hàm mục tiêu và gradient
        fval = f(x);
        grad = gradFunc(x);

        % Lưu trữ lịch sử
        history(iter, :) = x';
        fval_history(iter) = fval;

        % Xây dựng mô hình bậc hai
        B = eye(dim); % Ma trận Hessian đơn giản (có thể sử dụng các phương pháp xấp xỉ khác)
        m = @(p) fval + grad' * p + 0.5 * p' * B * p;

        % Giải bài toán con trong vùng tin cậy
        if norm(grad) > 0
            p = -Delta * grad / norm(grad); % Hướng đi đơn giản theo gradient
        else
            p = zeros(size(x));
        end

        % Tính toán giá trị hàm mục tiêu tại điểm mới
        x_new = x + p;
        fval_new = f(x_new);

        % Tính toán tỷ lệ cải thiện thực tế so với dự đoán
        rho = (fval - fval_new) / (m(zeros(size(x))) - m(p));

        % Cập nhật điểm hiện tại và bán kính vùng tin cậy
        if rho >= eta1
            x = x_new;
            best_solutions = [best_solutions, x];
            Delta = min(2 * Delta, 1.0); % Giới hạn tăng Delta
        elseif rho > eta2
            x = x_new;
            best_solutions = [best_solutions, x];
        else
            Delta = Delta / 2;
        end

        % Kiểm tra hội tụ
        if abs(fval - fval_new) < tolFun
            break;
        end
        if norm(p) < tolX
            break;
        end
    end

    % Kết thúc đếm thời gian và tính tổng thời gian chạy
    elapsed_time = toc;

    if length(initial_point) == 2
        
        % Create a grid of points for the contour plot
        [X, Y] = meshgrid(linspace(-5, 5, 100), linspace(-5, 5, 100));
        Z = arrayfun(@(x, y) f([x, y]), X, Y);
        
        % Plot the contour
        figure;
        contour(X, Y, Z, 200); % 50 contour levels
        hold on;
        
        % Plot the best solutions
        plot(best_solutions(1, :), best_solutions(2, :), 'r-o', 'LineWidth', 2, 'MarkerFaceColor', 'r');
        
        % Plot the last solution with a different color
        plot(target_point(1), target_point(2), 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
        
        xlabel('x');
        ylabel('y');
        title('Contour Plot with Best Solution Path');
        grid on;
        hold off;
    end
end

