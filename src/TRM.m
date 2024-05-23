function main()
    % Nhập liệu cho hàm mục tiêu
    fprintf('Hãy nhập hàm mục tiêu dưới dạng @ (x) <hàm của bạn>, ví dụ: @ (x) 100*(x(2)-x(1)^2)^2 + (1-x(1))^2 cho 2 biến hoặc @ (x) (x-1)^4 - 10*(x-1)^2 + 10*(x-1) cho 1 biến\n');
    objFuncStr = input('Nhập hàm mục tiêu: ', 's');
    objFunc = str2func(objFuncStr);

    % Nhập liệu cho gradient của hàm mục tiêu
    fprintf('Hãy nhập gradient của hàm mục tiêu dưới dạng @ (x) [<gradient của bạn>], ví dụ: @ (x) [-400*x(1)*(x(2)-x(1)^2) - 2*(1-x(1)); 200*(x(2)-x(1)^2)] cho 2 biến hoặc @ (x) 4*(x-1)^3 - 20*(x-1) + 10 cho 1 biến\n');
    gradFuncStr = input('Nhập gradient của hàm mục tiêu: ', 's');
    gradFunc = str2func(gradFuncStr);

    % Nhập liệu cho điểm khởi đầu
    fprintf('Nhập điểm khởi đầu dưới dạng vector hàng [x1, x2] cho 2 biến hoặc số thực x0 cho 1 biến\n');
    x0 = input('Nhập điểm khởi đầu: ');

    % Gọi hàm tối ưu hóa với hàm mục tiêu nhập vào
    options = struct('TolFun', 1e-6, 'TolX', 1e-6, 'MaxIter', 1000, 'Delta0', 1.0);
    [x, fval, exitflag, output] = trust_region_method(objFunc, gradFunc, x0, options);

    % Hiển thị kết quả
    if length(x) == 1
        fprintf('Solution: x = %f\n', x);
    else
        fprintf('Solution: x = [%f, %f]\n', x(1), x(2));
    end
    fprintf('Objective function value: fval = %f\n', fval);
    fprintf('Exit flag: %d\n', exitflag);
    fprintf('Number of iterations: %d\n', output.iterations);
    fprintf('Elapsed time: %.4f seconds\n', output.elapsedTime);

    % Vẽ đồ thị quỹ đạo của x1 và x2
    if length(x) == 1
        figure;
        x_range = linspace(min(output.history)-2, max(output.history)+2, 100);
        f_values = arrayfun(objFunc, x_range);
        plot(x_range, f_values, '-', 'LineWidth', 2);
        hold on;
        plot(output.history, output.fval_history, '-o', 'DisplayName', 'Solution Path', 'LineWidth', 2, 'MarkerSize', 6);
        xlabel('x');
        ylabel('f(x)');
        legend;
        title('Solution Path on the Objective Function');
        grid on;
        hold off;
    else
        figure;
        plot(output.history(:, 1), output.history(:, 2), '-o', 'DisplayName', 'Solution Path');
        xlabel('x1');
        ylabel('x2');
        legend;
        title('Solution Path of Optimization');

        % Tạo contour plot của hàm mục tiêu
        figure;
        x_range = linspace(min(output.history(:, 1))-1, max(output.history(:, 1))+1, 100);
        y_range = linspace(min(output.history(:, 2))-1, max(output.history(:, 2))+1, 100);
        [X, Y] = meshgrid(x_range, y_range);
        Z = arrayfun(@(x1, x2) objFunc([x1; x2]), X, Y);
        contour(X, Y, Z, 50);
        hold on;
        plot(output.history(:, 1), output.history(:, 2), '-o', 'DisplayName', 'Solution Path');
        xlabel('x1');
        ylabel('x2');
        legend;
        title('Contour Plot of Objective Function with Solution Path');
        grid on;
        hold off;
    end

    % Vẽ đồ thị giá trị hàm theo số lần lặp
    figure;
    plot(1:output.iterations, output.fval_history(1:output.iterations), '-o', 'DisplayName', 'Objective Function');
    xlabel('Iteration');
    ylabel('Objective Function Value');
    legend;
    title('Objective Function Value vs. Iteration');

end

function [x, fval, exitflag, output] = trust_region_method(objFunc, gradFunc, x0, options)
    % Bắt đầu đếm thời gian
    tic;

    % Thiết lập các thông số mặc định
    if nargin < 4
        options = struct();
    end
    if ~isfield(options, 'TolFun')
        options.TolFun = 1e-6;
    end
    if ~isfield(options, 'TolX')
        options.TolX = 1e-6;
    end
    if ~isfield(options, 'MaxIter')
        options.MaxIter = 1000;
    end
    if ~isfield(options, 'Delta0')
        options.Delta0 = 1.0;
    end

    % Khởi tạo
    x = x0(:); % Chuyển x0 thành vector cột
    Delta = options.Delta0;
    maxIter = options.MaxIter;
    tolFun = options.TolFun;
    tolX = options.TolX;
    eta1 = 0.75;
    eta2 = 0.25;
    iter = 0;
    exitflag = 0;

    % Lưu trữ lịch sử x và giá trị hàm mục tiêu
    history = zeros(maxIter, length(x));
    fval_history = zeros(maxIter, 1);

    while iter < maxIter
        iter = iter + 1;

        % Tính toán giá trị hàm mục tiêu và gradient
        fval = objFunc(x);
        grad = gradFunc(x);

        % Lưu trữ lịch sử
        history(iter, :) = x';
        fval_history(iter) = fval;

        % Xây dựng mô hình bậc hai
        B = eye(length(x)); % Ma trận Hessian đơn giản (có thể sử dụng các phương pháp xấp xỉ khác)
        m = @(p) fval + grad' * p + 0.5 * p' * B * p;

        % Giải bài toán con trong vùng tin cậy
        p = -B \ grad; % Hướng Newton đơn giản

        if norm(p) > Delta
            p = p * (Delta / norm(p)); % Điều chỉnh để nằm trong vùng tin cậy
        end

        % Tính toán giá trị hàm mục tiêu tại điểm mới
        x_new = x + p;
        fval_new = objFunc(x_new);

        % Tính toán tỷ lệ cải thiện thực tế so với dự đoán
        rho = (fval - fval_new) / (m(zeros(size(x))) - m(p));

        % Cập nhật điểm hiện tại và bán kính vùng tin cậy
        if rho >= eta1
            x = x_new;
            Delta = Delta * 2;
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
end

% Gọi hàm chính
main();
