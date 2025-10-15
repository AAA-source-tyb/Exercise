#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <complex>
#include <algorithm>

template <typename T>
class Matrix {
public:
    Matrix(int rows, int cols, bool owner = true)
        : rows_(rows), cols_(cols), data_(new T[rows * cols]), owner_(owner) {}
    
    Matrix(int rows, int cols, T* data, bool owner = false)
        : rows_(rows), cols_(cols), data_(data), owner_(owner) {
        if (data == nullptr) {
            throw std::invalid_argument("data is nullptr");
        }
    }
    
    Matrix(int rows, int cols, std::initializer_list<T> list)
        : rows_(rows), cols_(cols), data_(new T[rows * cols]), owner_(true) {
        if (list.size() != rows * cols) {
            throw std::invalid_argument("list size not match");
        }
        std::copy(list.begin(), list.end(), data_);
    }
    
    ~Matrix() {
        if (owner_ && data_ != nullptr) {
            delete[] data_;
        }
    }
    
    Matrix(const Matrix& other) {
        rows_ = other.rows_;
        cols_ = other.cols_;
        owner_ = true;
        data_ = new T[rows_ * cols_];
        std::copy(other.data_, other.data_ + rows_ * cols_, data_);
    }
    
    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            if (owner_ && data_ != nullptr) {
                delete[] data_;
            }
            rows_ = other.rows_;
            cols_ = other.cols_;
            owner_ = true;
            data_ = new T[rows_ * cols_];
            std::copy(other.data_, other.data_ + rows_ * cols_, data_);
        }
        return *this;
    }
    
    const int rows() const { return rows_; }
    const int cols() const { return cols_; }
    const T* data() const { return data_; }
    T* data() { return data_; }
    
    T& operator()(int row, int col) {
        if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
            throw std::out_of_range("Matrix indices out of range");
        }
        return data_[row * cols_ + col];
    }
    
    const T operator()(int row, int col) const {
        if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
            throw std::out_of_range("Matrix indices out of range");
        }
        return data_[row * cols_ + col];
    }
    
    // Matrix addition
    Matrix<T> operator+(const Matrix<T>& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrix size not match");
        }
        Matrix<T> result(rows_, cols_);
        for (int i = 0; i < rows_ * cols_; ++i) {
            result.data_[i] = data_[i] + other.data_[i];
        }
        return result;
    }
    
    // Matrix multiplication
    Matrix<T> operator*(const Matrix<T>& other) const {
        if (cols_ != other.rows_) {
            throw std::invalid_argument("Matrix size not match");
        }
        Matrix<T> result(rows_, other.cols_);
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < other.cols_; ++j) {
                T sum = T(0);
                for (int k = 0; k < cols_; ++k) {
                    sum += (*this)(i, k) * other(k, j);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }
    
    // Scalar multiplication
    Matrix<T> operator*(T scalar) const {
        Matrix<T> result(rows_, cols_);
        for (int i = 0; i < rows_ * cols_; ++i) {
            result.data_[i] = data_[i] * scalar;
        }
        return result;
    }
    
    friend std::ostream& operator<<(std::ostream& os, const Matrix<T>& mat) {
        for (int i = 0; i < mat.rows_; ++i) {
            for (int j = 0; j < mat.cols_; ++j) {
                os << mat(i, j) << " ";
            }
            os << "\n";
        }
        return os;
    }

private:
    int rows_, cols_;
    T* data_;
    bool owner_;
};

// Type traits for LAPACK
template <typename T> struct MatrixTypeInfo {
    using RealType = T;
};

template <> struct MatrixTypeInfo<std::complex<float>> {
    using RealType = float;
};

template <> struct MatrixTypeInfo<std::complex<double>> {
    using RealType = double;
};

// Eigen decomposition
template <typename T>
std::pair<std::vector<typename MatrixTypeInfo<T>::RealType>, Matrix<T>>
eigh(const Matrix<T>& M) {
    using RealType = typename MatrixTypeInfo<T>::RealType;
    
    if (M.rows() != M.cols()) {
        throw std::invalid_argument("Matrix must be square");
    }
    
    int n = M.rows();
    std::vector<RealType> eigenvalues(n);
    Matrix<T> eigenvectors(n, n);
    
    // Copy matrix data to eigenvectors (will be overwritten by LAPACK)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            eigenvectors(i, j) = M(i, j);
        }
    }
    
    // For this example, we'll use a simple Jacobi-like method for 2x2 matrix
    // In a real implementation, you would call LAPACK here
    
    if (n == 2) {
        // Simple 2x2 eigenvalue calculation
        T a = M(0, 0), b = M(0, 1), c = M(1, 0), d = M(1, 1);
        
        // For symmetric matrix: b should equal c
        T trace = a + d;
        T determinant = a * d - b * c;
        
        RealType discriminant = trace * trace - 4 * determinant;
        if (discriminant < 0) discriminant = 0; // For real symmetric matrices
        
        eigenvalues[0] = (trace - std::sqrt(discriminant)) / 2;
        eigenvalues[1] = (trace + std::sqrt(discriminant)) / 2;
        
        // Eigenvectors
        if (std::abs(b) > 1e-10) {
            eigenvectors(0, 0) = eigenvalues[0] - d;
            eigenvectors(1, 0) = b;
            eigenvectors(0, 1) = eigenvalues[1] - d;
            eigenvectors(1, 1) = b;
        } else {
            eigenvectors(0, 0) = 1; eigenvectors(1, 0) = 0;
            eigenvectors(0, 1) = 0; eigenvectors(1, 1) = 1;
        }
        
        // Normalize eigenvectors
        for (int j = 0; j < 2; ++j) {
            RealType norm = std::sqrt(std::norm(eigenvectors(0, j)) + 
                                     std::norm(eigenvectors(1, j)));
            if (norm > 1e-10) {
                eigenvectors(0, j) /= norm;
                eigenvectors(1, j) /= norm;
            }
        }
    }
    
    return {eigenvalues, eigenvectors};
}

#endif