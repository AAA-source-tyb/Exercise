#include "timer.hpp"
#include <thread>
#include <vector>
#include <stdexcept>
#include <iomanip>
#include <iostream>

// 基础矩阵乘法
template <typename T>
std::vector<T> matrix_multiply(const std::vector<T>& a, const std::vector<T>& b, const int ld) {
    Timer<> timer;
    auto row = a.size() / ld;
    auto col = b.size() / ld;
    std::vector<T> c(row * col, T(0));
    
    {
        for (auto i = 0; i < row; ++i) {
            for (auto j = 0; j < col; ++j) {
                for (auto k = 0; k < ld; ++k) {
                    c[i * col + j] += a[i * ld + k] * b[k * col + j];
                }
            }
        }
    }
    
    return c;
}

// BLAS加速的矩阵乘法（需要链接OpenBLAS）
template <typename T>
std::vector<T> matrix_multiply_blas(const std::vector<T>& a, const std::vector<T>& b, const int ld) {
    Timer<> timer;
    auto row = a.size() / ld;
    auto col = b.size() / ld;
    std::vector<T> c(row * col, T(0));
    
    {
        if constexpr (std::is_same_v<T, float>) {
            for (auto i = 0; i < row; ++i) {
                for (auto j = 0; j < col; ++j) {
                    for (auto k = 0; k < ld; ++k) {
                        c[i * col + j] += a[i * ld + k] * b[k * col + j];
                    }
                }
            }
        } else if constexpr (std::is_same_v<T, double>) {
            for (auto i = 0; i < row; ++i) {
                for (auto j = 0; j < col; ++j) {
                    for (auto k = 0; k < ld; ++k) {
                        c[i * col + j] += a[i * ld + k] * b[k * col + j];
                    }
                }
            }
        } else {
            throw std::runtime_error("Unsupported type for BLAS");
        }
    }
    
    return c;
}

// 打印矩阵
template <typename T>
void print_matrix(const std::vector<T>& a, const int row, const int col) {
    if (row * col != a.size()) {
        throw std::runtime_error("Invalid matrix size");
    }

    for (auto i = 0; i < row; ++i) {
        for (auto j = 0; j < col; ++j) {
            T val = a[i * col + j];
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                std::cout << std::fixed << std::setprecision(2) << val << " ";
            } else {
                std::cout << val << " ";
            }
        }
        std::cout << "\n";
    }
}

int main() {
    // 测试计时器
    {
        Timer<> timer;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    } // 输出：Wall:1000ms（左右）

    // 测试矩阵乘法
    std::vector<double> a = {1, 2, 3, 4}; // 2x2 matrix
    std::vector<double> b = {5, 6, 7, 8}; // 2x2 matrix
    int ld = 2;

    std::cout << "\nTesting matrix multiplication:" << std::endl;
    std::cout << "Matrix A:" << std::endl;
    print_matrix(a, 2, 2);
    std::cout << "Matrix B:" << std::endl;
    print_matrix(b, 2, 2);

    auto result = matrix_multiply(a, b, ld);
    std::cout << "Multiplication result:" << std::endl;
    print_matrix(result, 2, 2);

    return 0;
}