function [x, fval, exitflag, output] = trust_region_method(f, x0)
    % Thiết lập các thông số mặc định nếu không được truyền vào
    defaultOptions = struct('TolFun', 1e-8, 'TolX', 1e-8, 'MaxIter', 10000, 'Delta0', 1.0);
    if nargin < 3
        options = defaultOptions;
    end

    % Bắt đầu đếm thời gian
    tic;

    % Khởi tạo
    x_sym = sym('x', [1, length(x0)]); % Biến symbolic
    x = x0(:); % Chuyển x0 thành vector cột
    Delta = options.Delta0;
    maxIter = options.MaxIter;
    tolFun = options.TolFun;
    tolX = options.TolX;
    eta1 = 0.75;
    eta2 = 0.25;
    iter = 0;
    exitflag = 0;

    dim = length(x0);

    % Tính gradient của hàm mục tiêu
    gradf = gradient(f(x_sym), x_sym);

    % Chuyển hàm symbolic gradient thành hàm số
    gradFunc = matlabFunction(gradf, 'Vars', {x_sym});

    % Lưu trữ lịch sử x và giá trị hàm mục tiêu
    history = zeros(maxIter, dim);
    fval_history = zeros(maxIter, 1);

    while iter < maxIter
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
            Delta = min(2 * Delta, 1.0); % Giới hạn tăng Delta
        elseif rho > eta2
            x = x_new;
        else
            Delta = Delta / 2;
        end

        % Kiểm tra hội tụ
        if abs(fval - fval_new) < tolFun
            exitflag = 1;
            break;
        end
        if norm(p) < tolX
            exitflag = 2;
            break;
        end
    end

    % Kết thúc đếm thời gian và tính tổng thời gian chạy
    elapsedTime = toc;

    % Kết quả
    output.iterations = iter;
    output.message = 'Optimization terminated successfully.';
    output.history = history(1:iter, :);
    output.fval_history = fval_history(1:iter);
    output.elapsedTime = elapsedTime;

    % Hiển thị kết quả
    disp(['Solution: x = ', num2str(x)]);
    disp(['Objective function value: fval = ', num2str(fval)]);
    disp(['Exit flag: ', num2str(exitflag)]);
    disp(['Number of iterations: ', num2str(iter)]);
    disp(['Elapsed time: ', num2str(elapsedTime), ' seconds']);

    % Vẽ đồ thị giá trị hàm theo số lần lặp
    figure;
    plot(1:output.iterations, output.fval_history(1:output.iterations), '-o', 'DisplayName', 'Objective Function');
    xlabel('Iteration');
    ylabel('Objective Function Value');
    legend;
    title('Objective Function Value vs. Iteration');
end

% Ví dụ gọi hàm trust_region_method từ bên ngoài
objFunc = @(x) 100*(x(2)-x(1)^2)^2 + (1-x(1))^2;
x0 = [-1.2, 1];
[x, fval, exitflag, output] = trust_region_method(objFunc, x0);
