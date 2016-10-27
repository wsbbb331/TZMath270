namespace JIXIE {
    template <class TA, class TU, class Ts, class TV>
    inline std::enable_if_t<isSize<TA>(2, 2) && isSize<TU>(2, 2) && isSize<TV>(2, 2) && isSize<Ts>(2, 2)>
    My_SVD(
                               const Eigen::MatrixBase<TA>& A,
                               const Eigen::MatrixBase<TU>& U,
                               const Eigen::MatrixBase<Ts>& Sigma,
                               const Eigen::MatrixBase<TV>& V,
                               const ScalarType<TA> tol = 64 * std::numeric_limits<ScalarType<TA> >::epsilon())
    {
        using std::sqrt;
        using T = ScalarType<TA>;
        Eigen::MatrixBase<Ts>& sigma = const_cast<Eigen::MatrixBase<Ts>&>(Sigma);
        Eigen::MatrixBase<TU>& u = const_cast<Eigen::MatrixBase<TU>&>(U);
        Eigen::MatrixBase<TV>& v = const_cast<Eigen::MatrixBase<TV>&>(V);
        Eigen::Matrix<T, 2, 1> sigmaHat;
        bool det_v_negative = false;
        bool det_u_negative = false;
        
        Eigen::Matrix<T, 2, 2> C;
        C.noalias() = A.transpose() * A;
        T cosine, sine;
        T x = C(0, 0);
        T y = C(0, 1);
        T z = C(1, 1);
        if (y == 0) {
            // C is already diagonal
            cosine = 1;
            sine = 0;
            sigmaHat(0) = x;
            sigmaHat(1) = z;
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
            sigmaHat(0) = c2 * x - csy + s2 * z;
            sigmaHat(1) = s2 * x + csy + c2 * z;
        }
        
        GivensRotation<T> VHat(0,1);
        
        if (sigmaHat(0) < sigmaHat(1)) {
            std::swap(sigmaHat(0), sigmaHat(1));
            VHat.c = -sine;
            VHat.s = cosine;
            det_v_negative = true;
        }
        else {
            VHat.c = cosine;
            VHat.s = sine;
        }
        
        Eigen::Matrix<T, 2, 2> VHatMatrix;
        VHat.fill(VHatMatrix);
        
        sigmaHat(0) = sqrt(sigmaHat(0));
        sigmaHat(1) = sqrt(sigmaHat(1));
        
        T Adet = A.determinant();
        Eigen::Matrix<T, 2, 2> F = A;

        VHat.columnRotation(F);
        
        GivensRotation<T> Q(0,1);
        Q.compute(F(0,0), F(1,0));
        
        T r22 = Q.s * F(0,1) + Q.c * F(1,1);
        Eigen::Matrix<T, 2, 2> UHat;
        if (r22 < 0){
            Q.fill(UHat);
            UHat(0,1) = -UHat(0,1);
            UHat(1,1) = -UHat(1,1);
            det_u_negative = true;
        }
        else{
            Q.fill(UHat);
        }
        
        if (Adet < 0){
            sigma(0, 0) = sigmaHat(0);
            sigma(1, 1) = -sigmaHat(1);
            if(det_u_negative){
                u.col(1) = -UHat.col(1);
                u.col(0) = UHat.col(0);
                v.col(1) = VHatMatrix.col(1);
                v.col(0) = VHatMatrix.col(0);
            }
            else{
                u.col(1) = UHat.col(1);
                u.col(0) = UHat.col(0);
                v.col(1) = -VHatMatrix.col(1);
                v.col(0) = VHatMatrix.col(0);
            }
        }
        else if(Adet > 0){
            sigma(0, 0) = sigmaHat(0);
            sigma(1, 1) = sigmaHat(1);
            if(det_u_negative && det_v_negative){
                u.col(1) = -UHat.col(1);
                u.col(0) = UHat.col(0);
                v.col(1) = -VHatMatrix.col(1);
                v.col(0) = VHatMatrix.col(0);
            }
            else{
                u.col(1) = UHat.col(1);
                u.col(0) = UHat.col(0);
                v.col(1) = VHatMatrix.col(1);
                v.col(0) = VHatMatrix.col(0);
            }
        }
        else if(Adet == 0){
            sigma(0, 0) = sigmaHat(0);
            sigma(1, 1) = sigmaHat(1);
            u.col(1) = UHat.col(1);
            u.col(0) = UHat.col(0);
            v.col(1) = VHatMatrix.col(1);
            v.col(0) = VHatMatrix.col(0);
        }
        
    }
    
}


