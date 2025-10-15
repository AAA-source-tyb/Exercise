#include "matrix.hpp"
#include <iostream>
#include <vector>

int main() {
   
    std::vector<double> data = {0.0, 1.0, 1.0, 0.0};
    Matrix<double> H(2, 2, data.data(), false);
    
    std::cout << "Hückel Matrix H:" << std::endl;
    std::cout << H << std::endl;
    
    auto [eigenvalues, eigenvectors] = eigh(H);
    
    std::cout << "Eigenvalues (Energy levels):" << std::endl;
    for (size_t i = 0; i < eigenvalues.size(); ++i) {
        std::cout << "E" << i << " = " << eigenvalues[i] << std::endl;
    }
    
    std::cout << "\nEigenvectors (Molecular orbitals):" << std::endl;
    std::cout << eigenvectors << std::endl;
    
    return 0;
}