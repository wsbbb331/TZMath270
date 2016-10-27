//
//  algorithm3.h
//  Math270AHW1
//
//  Created by Tianyang Zhang on 10/14/16.
//  Copyright © 2016 UCLA. All rights reserved.
//

namespace JIXIE {
    
    template<class T>
    inline void jacobiRotation(Eigen::Matrix<T, 2, 1>& sigma,
                               GivensRotation<T>& V,
                               Eigen::Matrix<T, 2, 2>& S_Sym){
        T cosine, sine;
        T x = S_Sym(0, 0);
        T y = S_Sym(0, 1);
        T z = S_Sym(1, 1);
        if (y == 0) {
            // S is already diagonal
            cosine = 1;
            sine = 0;
            sigma(0) = x;
            sigma(1) = z;
        }
        else {
            T tau = 0.5 * (x - z);
            T w = sqrt(tau * tau + y * y);
            // w > y > 0
            T t;
            if (tau > 0) {
                // tau + w > w > y > 0 ==> division is safe
                t = y / (tau + w);
            }
            else {
                // tau - w < -w < -y < 0 ==> division is safe
                t = y / (tau - w);
            }
            cosine = T(1) / sqrt(t * t + T(1));
            sine = -t * cosine;
            /*
             V = [cosine -sine; sine cosine]
             Sigma = V'SV. Only compute the diagonals for efficiency.
             Also utilize symmetry of S and don't form V yet.
             */
            T c2 = cosine * cosine;
            T csy = 2 * cosine * sine * y;
            T s2 = sine * sine;
            sigma(0) = c2 * x - csy + s2 * z;
            sigma(1) = s2 * x + csy + c2 * z;
        }

        // Sorting
        // Polar already guarantees negative sign is on the small magnitude singular value.
        if (sigma(0) < sigma(1)) {
            std::swap(sigma(0), sigma(1));
            V.c = -sine;
            V.s = cosine;
        }
        else {
            V.c = cosine;
            V.s = sine;
        }
    }
    
    template <class T>
    inline void polarDecomposition(const Eigen::Matrix<T, 3, 3>& A,
        Eigen::Matrix<T, 3, 3>& R,
        Eigen::Matrix<T, 3, 3>& S_Sym,
        T tol = 128 * std::numeric_limits<T>::epsilon())
    {
        using std::sqrt;
        using std::fabs;
        using std::pow;
        std::string sep = "\n----------------------------------------\n";
        
        S_Sym = A; //S symmetric
        Eigen::Matrix<T, 2, 2> S_Sym22;//2x2 placeholder
        Eigen::Matrix<T, 2, 2> A22;//part of A
        GivensRotation<T> R22(0, 1);//Iterative R
        R = Eigen::Matrix<T, 3, 3>::Identity();
        
        int count = 0;
        int maxcount = pow(10,3);
        while((fabs(S_Sym(0,1) - S_Sym(1,0)) > tol || fabs(S_Sym(0,2) - S_Sym(2,0)) > tol || fabs(S_Sym(1,2) - S_Sym(2,1)) > tol) && count <= maxcount){
            
            R22.rowi = 0;
            R22.rowk = 1;
            
            if(count % 3 == 0){
                A22 = S_Sym.topLeftCorner(2,2);
                polarDecomposition(A22, R22, S_Sym22);
            }
            else if(count % 3 == 1){
                A22 << S_Sym(0,0), S_Sym(0,2), S_Sym(2,0), S_Sym(2,2);
                polarDecomposition(A22, R22, S_Sym22);
                R22.rowi = 0;
                R22.rowk = 2;
            }
            else if(count % 3 == 2){
                A22 = S_Sym.bottomRightCorner(2,2);
                polarDecomposition(A22, R22, S_Sym22);
                R22.rowi = 1;
                R22.rowk = 2;
            }
            
            R22.rowRotation(S_Sym);
//            std::cout << S_Sym << sep;
            R22.columnRotation(R);
            count += 1;
        }
        std::cout << count << sep;
    }
    
    
    template <class T>
    inline void singularValueDecomposition(const Eigen::Matrix<T, 3, 3>& A,
                                          Eigen::Matrix<T, 3, 3>& U,
                                          Eigen::Matrix<T, 3, 1>& sigma,
                                          Eigen::Matrix<T, 3, 3>& V,
                                           T tol = 128 * std::numeric_limits<T>::epsilon())
    {
        using std::sqrt;
        using std::fabs;
        using std::pow;
        std::string sep = "\n----------------------------------------\n";

        Eigen::Matrix<T, 3, 3> S = A; //S symmetric
        Eigen::Matrix<T, 2, 2> S_Sym;//2x2 placeholder
        Eigen::Matrix<T, 2, 2> A22;//part of A
        GivensRotation<T> R22(0, 1);//Iterative R
        U = Eigen::Matrix<T, 3, 3>::Identity();// R
        
        int count = 0;
        int maxcount = pow(10,5);
        while((fabs(S(0,1) - S(1,0)) > tol || fabs(S(0,2) - S(2,0)) > tol || fabs(S(1,2) - S(2,1)) > tol) && count <= maxcount){

            R22.rowi = 0;
            R22.rowk = 1;
            
            if(count % 3 == 0){
                A22 = S.topLeftCorner(2,2);
                polarDecomposition(A22, R22, S_Sym);
                //            R12.rowi = 0;
                //            R12.rowk = 1;
            }
            else if(count % 3 == 1){
                A22 << S(0,0), S(0,2), S(2,0), S(2,2);
                polarDecomposition(A22, R22, S_Sym);
                R22.rowi = 0;
                R22.rowk = 2;
            }
            else if(count % 3 == 2){
                A22 = S.bottomRightCorner(2,2);
                polarDecomposition(A22, R22, S_Sym);
                R22.rowi = 1;
                R22.rowk = 2;
            }
            
            R22.rowRotation(S);
//            std::cout << S << sep;
            R22.columnRotation(U);
            count += 1;
        }
        
        std::cout << count << sep;
        
        GivensRotation<T> V22(0, 1);//iterative V
        V = Eigen::Matrix<T, 3, 3>::Identity();
        Eigen::Matrix<T, 2, 2> S22;//part of S
        Eigen::Matrix<T, 2, 1> sigma21;//placeholder of sigma
        
        count = 0;
        while((fabs(S(0,1)) > tol || fabs(S(0,2)) > tol || fabs(S(1,2)) > tol || fabs(S(1,0)) > tol || fabs(S(2,0)) > tol || fabs(S(2,1)) > tol) && count <= maxcount){
            V22.rowi = 0;
            V22.rowk = 1;
            
            if (count % 3 == 0){
                S22 = S.topLeftCorner(2,2);
                jacobiRotation(sigma21, V22, S22);
                V22.rowRotation(S);
                V22.columnRotation(S);
//                std::cout << S << sep;
            }
            else if (count % 3 == 1){
                S22 << S(0,0), S(0,2), S(2,0), S(2,2);
                jacobiRotation(sigma21, V22, S22);
                V22.rowi = 0;
                V22.rowk = 2;
                V22.rowRotation(S);
                V22.columnRotation(S);
//                std::cout << S << sep;
                
            }
            else if (count % 3 == 2){
                S22 = S.bottomRightCorner(2,2);
                jacobiRotation(sigma21, V22, S22);
                V22.rowi = 1;
                V22.rowk = 2;
                V22.rowRotation(S);
                V22.columnRotation(S);
//                std::cout << S << sep;
            }
            
            V22.columnRotation(V);
            count += 1;
        }
        std::cout << count << sep;
        
        sigma << S(0,0), S(1,1), S(2,2);
        
        U = U * V;
        
    }
}
