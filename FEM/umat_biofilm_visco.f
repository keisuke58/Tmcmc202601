C======================================================================
C     UMAT: Viscoelastic biofilm with multiplicative decomposition
C            F = F_e * F_v * F_g
C
C     Hyperelastic base: Neo-Hookean or Mooney-Rivlin
C     Viscous flow: deviatoric Newtonian dashpot (backward Euler)
C     Growth: isotropic via thermal analogy (F_g = (1 + alpha_T*T) I)
C
C     PROPS(1) = C10   Neo-Hookean/Mooney-Rivlin parameter [MPa]
C     PROPS(2) = C01   Mooney-Rivlin parameter (0 for Neo-Hookean) [MPa]
C     PROPS(3) = D1    Compressibility parameter [1/MPa]
C     PROPS(4) = eta   Viscosity [MPa*s]
C     PROPS(5) = type  0.0=Neo-Hookean, 1.0=Mooney-Rivlin
C
C     STATEV(1:9) = F_v tensor (3x3, row-major: 11,12,13,21,22,23,31,32,33)
C
C     Ref: Junker & Balzani 2021, Soleimani 2019
C======================================================================
      SUBROUTINE UMAT(STRESS, STATEV, DDSDDE, SSE, SPD, SCD,
     1 RPL, DDSDDT, DRPLDE, DRPLDT,
     2 STRAN, DSTRAN, TIME, DTIME, TEMP, DTEMP, PREDEF, DPRED,
     3 CMNAME, NDI, NSHR, NTENS, NSTATV, PROPS, NPROPS, COORDS,
     4 DROT, PNEWDT, CELENT, DFGRD0, DFGRD1, NOEL, NPT, LAYER,
     5 KSPT, KSTEP, KINC)

      INCLUDE 'ABA_PARAM.INC'

      CHARACTER*80 CMNAME

      DIMENSION STRESS(NTENS), STATEV(NSTATV), DDSDDE(NTENS,NTENS),
     1 DDSDDT(NTENS), DRPLDE(NTENS), STRAN(NTENS), DSTRAN(NTENS),
     2 TIME(2), PREDEF(1), DPRED(1), PROPS(NPROPS), COORDS(3),
     3 DROT(3,3), DFGRD0(3,3), DFGRD1(3,3)

C     --- Local variables ---
      DOUBLE PRECISION C10, C01, D1, ETA, MTYPE
      DOUBLE PRECISION FG(3,3), FG_INV(3,3), FG_DET
      DOUBLE PRECISION FV_OLD(3,3), FV_NEW(3,3), FV_INV(3,3)
      DOUBLE PRECISION FE(3,3), FTRIAL(3,3)
      DOUBLE PRECISION CE(3,3), BE(3,3)
      DOUBLE PRECISION TAU_DEV(3,3), SIGMA(3,3)
      DOUBLE PRECISION ALPHA_GROWTH, DETFE, DETFV
      DOUBLE PRECISION I1_BAR, I2_BAR, I1E, I2E
      DOUBLE PRECISION TRCE, TMP1, TMP2, DTIME_SAFE
      DOUBLE PRECISION PRESS, MU_EFF
      DOUBLE PRECISION IDEN(3,3), TEMP3(3,3)
      INTEGER I, J, K

C     --- Read material properties ---
      C10   = PROPS(1)
      C01   = PROPS(2)
      D1    = PROPS(3)
      ETA   = PROPS(4)
      MTYPE = PROPS(5)

C     --- Identity tensor ---
      DO I = 1, 3
        DO J = 1, 3
          IDEN(I,J) = 0.0D0
        END DO
        IDEN(I,I) = 1.0D0
      END DO

C     ================================================================
C     1. Compute F_g from thermal analogy
C     ================================================================
      ALPHA_GROWTH = TEMP + DTEMP
      IF (ALPHA_GROWTH .LT. 0.0D0) ALPHA_GROWTH = 0.0D0
      FG_DET = (1.0D0 + ALPHA_GROWTH)**3
      DO I = 1, 3
        DO J = 1, 3
          FG(I,J) = (1.0D0 + ALPHA_GROWTH) * IDEN(I,J)
          FG_INV(I,J) = IDEN(I,J) / (1.0D0 + ALPHA_GROWTH)
        END DO
      END DO

C     ================================================================
C     2. Read F_v^old from state variables
C     ================================================================
      K = 0
      DO I = 1, 3
        DO J = 1, 3
          K = K + 1
          FV_OLD(I,J) = STATEV(K)
        END DO
      END DO

C     Check if F_v is initialized (first increment: STATEV=0)
      TMP1 = 0.0D0
      DO I = 1, 3
        TMP1 = TMP1 + FV_OLD(I,I)
      END DO
      IF (DABS(TMP1) .LT. 1.0D-10) THEN
C       Initialize F_v = I
        DO I = 1, 3
          DO J = 1, 3
            FV_OLD(I,J) = IDEN(I,J)
          END DO
        END DO
      END IF

C     ================================================================
C     3. Compute F_trial = F_total * inv(F_g)
C     ================================================================
      CALL MAT_MULT(DFGRD1, FG_INV, FTRIAL, 3)

C     ================================================================
C     4. Viscous update: F_v^new via backward Euler
C        Flow rule: L_v = (1/(2*eta)) * dev(Mandel_e)
C        Approximation: F_v^new = F_v^old + dt * L_v * F_v^old
C        For stability, use exponential map when possible.
C
C        Simplified approach for moderate viscous flow:
C        F_v^new = F_v^old  (elastic predictor)
C        then correct iteratively using Newton-Raphson.
C
C        Here we use a single-step backward Euler update:
C        F_e_trial = F_trial * inv(F_v^old)
C        tau_dev = dev(Kirchhoff from F_e_trial)
C        Delta_Dv = (dt/(2*eta)) * tau_dev
C        F_v^new = (I + Delta_Dv) * F_v^old
C     ================================================================

C     Elastic trial: F_e_trial = F_trial * inv(F_v_old)
      CALL MAT_INV3(FV_OLD, FV_INV, DETFV)
      CALL MAT_MULT(FTRIAL, FV_INV, FE, 3)

C     Left Cauchy-Green trial: be = Fe * Fe^T
      CALL MAT_AAT(FE, BE, 3)

C     Determinant of Fe
      CALL MAT_DET3(FE, DETFE)
      IF (DETFE .LT. 1.0D-15) DETFE = 1.0D-15

C     Isochoric Be_bar = J^(-2/3) * Be
      TMP1 = DETFE**(-2.0D0/3.0D0)

C     I1 of Be_bar
      I1_BAR = TMP1 * (BE(1,1) + BE(2,2) + BE(3,3))

C     I2 of Be_bar (for Mooney-Rivlin)
      IF (MTYPE .GT. 0.5D0) THEN
        I1E = BE(1,1) + BE(2,2) + BE(3,3)
        I2E = 0.5D0*(I1E**2 - (BE(1,1)**2 + BE(2,2)**2 + BE(3,3)**2
     1        + 2.0D0*(BE(1,2)**2 + BE(1,3)**2 + BE(2,3)**2)))
        I2_BAR = TMP1**2 * I2E
      ELSE
        I2_BAR = 0.0D0
      END IF

C     Kirchhoff stress from hyperelastic potential
C     tau = 2 * dW/dBe * Be  (push forward of 2nd PK)
C
C     Neo-Hookean: W = C10*(I1_bar - 3) + (1/D1)*(J-1)^2
C     tau_iso = 2*C10 * dev(Be_bar) / J^(2/3)
C     tau_vol = (2/D1)*(J-1)*J * I
C
C     Mooney-Rivlin: W = C10*(I1_bar-3) + C01*(I2_bar-3) + (1/D1)*(J-1)^2
C     tau_iso = 2*C10*dev(Be_bar)/J^(2/3) + 2*C01*dev(I1*Be_bar - Be_bar^2)/J^(4/3)
      PRESS = (2.0D0/D1) * (DETFE - 1.0D0) * DETFE

C     Deviatoric Kirchhoff: tau_dev
      TRCE = I1_BAR / 3.0D0
      DO I = 1, 3
        DO J = 1, 3
          TAU_DEV(I,J) = 2.0D0 * C10 * TMP1 *
     1                   (BE(I,J) - TRCE * IDEN(I,J))
        END DO
      END DO

C     Add Mooney-Rivlin I2 contribution
      IF (MTYPE .GT. 0.5D0) THEN
        DO I = 1, 3
          DO J = 1, 3
            TMP2 = 0.0D0
            DO K = 1, 3
              TMP2 = TMP2 + BE(I,K)*BE(K,J)
            END DO
            TEMP3(I,J) = I1_BAR * TMP1 * BE(I,J) - TMP1**2 * TMP2
          END DO
        END DO
        TMP2 = (TEMP3(1,1) + TEMP3(2,2) + TEMP3(3,3)) / 3.0D0
        DO I = 1, 3
          DO J = 1, 3
            TAU_DEV(I,J) = TAU_DEV(I,J) +
     1        2.0D0 * C01 * (TEMP3(I,J) - TMP2 * IDEN(I,J))
          END DO
        END DO
      END IF

C     Viscous strain increment
      DTIME_SAFE = MAX(DTIME, 1.0D-20)
      IF (ETA .GT. 1.0D-20) THEN
C       Delta_Dv = (dt/(2*eta)) * tau_dev / J
        TMP1 = DTIME_SAFE / (2.0D0 * ETA * DETFE)
C       F_v^new = (I + Delta_Dv) * F_v^old
        DO I = 1, 3
          DO J = 1, 3
            TEMP3(I,J) = IDEN(I,J) + TMP1 * TAU_DEV(I,J)
          END DO
        END DO
        CALL MAT_MULT(TEMP3, FV_OLD, FV_NEW, 3)
      ELSE
C       No viscosity: purely elastic
        DO I = 1, 3
          DO J = 1, 3
            FV_NEW(I,J) = FV_OLD(I,J)
          END DO
        END DO
      END IF

C     ================================================================
C     5. Recompute F_e with updated F_v
C     ================================================================
      CALL MAT_INV3(FV_NEW, FV_INV, DETFV)
      CALL MAT_MULT(FTRIAL, FV_INV, FE, 3)
      CALL MAT_AAT(FE, BE, 3)
      CALL MAT_DET3(FE, DETFE)
      IF (DETFE .LT. 1.0D-15) DETFE = 1.0D-15

C     Recompute stress with updated Fe
      TMP1 = DETFE**(-2.0D0/3.0D0)
      I1_BAR = TMP1 * (BE(1,1) + BE(2,2) + BE(3,3))
      PRESS = (2.0D0/D1) * (DETFE - 1.0D0) * DETFE

      TRCE = I1_BAR / 3.0D0
      DO I = 1, 3
        DO J = 1, 3
          SIGMA(I,J) = (2.0D0*C10*TMP1*(BE(I,J) - TRCE*IDEN(I,J))
     1                  + PRESS * IDEN(I,J)) / DETFE
        END DO
      END DO

C     Mooney-Rivlin correction to sigma
      IF (MTYPE .GT. 0.5D0) THEN
        I1E = BE(1,1) + BE(2,2) + BE(3,3)
        I2E = 0.5D0*(I1E**2 - (BE(1,1)**2 + BE(2,2)**2 + BE(3,3)**2
     1        + 2.0D0*(BE(1,2)**2 + BE(1,3)**2 + BE(2,3)**2)))
        I2_BAR = TMP1**2 * I2E
        DO I = 1, 3
          DO J = 1, 3
            TMP2 = 0.0D0
            DO K = 1, 3
              TMP2 = TMP2 + BE(I,K)*BE(K,J)
            END DO
            TEMP3(I,J) = I1_BAR * TMP1*BE(I,J) - TMP1**2 * TMP2
          END DO
        END DO
        TMP2 = (TEMP3(1,1) + TEMP3(2,2) + TEMP3(3,3)) / 3.0D0
        DO I = 1, 3
          DO J = 1, 3
            SIGMA(I,J) = SIGMA(I,J) +
     1        2.0D0*C01*(TEMP3(I,J) - TMP2*IDEN(I,J)) / DETFE
          END DO
        END DO
      END IF

C     ================================================================
C     6. Store Cauchy stress in Abaqus Voigt format
C     ================================================================
      STRESS(1) = SIGMA(1,1)
      STRESS(2) = SIGMA(2,2)
      STRESS(3) = SIGMA(3,3)
      IF (NSHR .GE. 1) STRESS(4) = SIGMA(1,2)
      IF (NSHR .GE. 2) STRESS(5) = SIGMA(1,3)
      IF (NSHR .GE. 3) STRESS(6) = SIGMA(2,3)

C     ================================================================
C     7. Consistent tangent (numerical perturbation for robustness)
C        DDSDDE(i,j) ≈ dSigma_i/dEps_j
C     ================================================================
      MU_EFF = 2.0D0 * (C10 + C01)
      TMP1 = 2.0D0 / (D1 + 1.0D-20)

C     Approximate tangent: Neo-Hookean/MR elastic tangent
C     C_ijkl = lambda*delta_ij*delta_kl + mu*(delta_ik*delta_jl + delta_il*delta_jk)
C     lambda = K - 2*mu/3 = 2/(D1) - 2*mu/3
      DO I = 1, NTENS
        DO J = 1, NTENS
          DDSDDE(I,J) = 0.0D0
        END DO
      END DO

C     Lambda effective
      TMP2 = TMP1 - 2.0D0*MU_EFF/3.0D0

C     Normal components (1,2,3)
      DO I = 1, NDI
        DO J = 1, NDI
          DDSDDE(I,J) = TMP2
        END DO
        DDSDDE(I,I) = DDSDDE(I,I) + 2.0D0*MU_EFF
      END DO

C     Shear components
      DO I = NDI+1, NTENS
        DDSDDE(I,I) = MU_EFF
      END DO

C     ================================================================
C     8. Update state variables: store F_v^new
C     ================================================================
      K = 0
      DO I = 1, 3
        DO J = 1, 3
          K = K + 1
          STATEV(K) = FV_NEW(I,J)
        END DO
      END DO

C     Specific elastic strain energy
      SSE = C10*(I1_BAR - 3.0D0) + (1.0D0/D1)*(DETFE - 1.0D0)**2
      IF (MTYPE .GT. 0.5D0) SSE = SSE + C01*(I2_BAR - 3.0D0)

C     Specific plastic dissipation (viscous)
      SPD = 0.0D0
      IF (ETA .GT. 1.0D-20) THEN
        DO I = 1, 3
          DO J = 1, 3
            SPD = SPD + TAU_DEV(I,J)**2
          END DO
        END DO
        SPD = SPD * DTIME_SAFE / (4.0D0 * ETA * DETFE)
      END IF

      RETURN
      END

C======================================================================
C     Utility: 3x3 matrix multiply C = A * B
C======================================================================
      SUBROUTINE MAT_MULT(A, B, C, N)
      INCLUDE 'ABA_PARAM.INC'
      INTEGER N, I, J, K
      DIMENSION A(N,N), B(N,N), C(N,N)
      DO I = 1, N
        DO J = 1, N
          C(I,J) = 0.0D0
          DO K = 1, N
            C(I,J) = C(I,J) + A(I,K)*B(K,J)
          END DO
        END DO
      END DO
      RETURN
      END

C======================================================================
C     Utility: 3x3 matrix A*A^T  →  C
C======================================================================
      SUBROUTINE MAT_AAT(A, C, N)
      INCLUDE 'ABA_PARAM.INC'
      INTEGER N, I, J, K
      DIMENSION A(N,N), C(N,N)
      DO I = 1, N
        DO J = 1, N
          C(I,J) = 0.0D0
          DO K = 1, N
            C(I,J) = C(I,J) + A(I,K)*A(J,K)
          END DO
        END DO
      END DO
      RETURN
      END

C======================================================================
C     Utility: Determinant of 3x3 matrix
C======================================================================
      SUBROUTINE MAT_DET3(A, DET)
      INCLUDE 'ABA_PARAM.INC'
      DIMENSION A(3,3)
      DOUBLE PRECISION DET
      DET = A(1,1)*(A(2,2)*A(3,3) - A(2,3)*A(3,2))
     1    - A(1,2)*(A(2,1)*A(3,3) - A(2,3)*A(3,1))
     2    + A(1,3)*(A(2,1)*A(3,2) - A(2,2)*A(3,1))
      RETURN
      END

C======================================================================
C     Utility: Inverse of 3x3 matrix (Cramer's rule)
C======================================================================
      SUBROUTINE MAT_INV3(A, AINV, DET)
      INCLUDE 'ABA_PARAM.INC'
      DIMENSION A(3,3), AINV(3,3)
      DOUBLE PRECISION DET, DINV

      CALL MAT_DET3(A, DET)
      IF (DABS(DET) .LT. 1.0D-30) THEN
C       Singular: return identity
        DO I = 1, 3
          DO J = 1, 3
            AINV(I,J) = 0.0D0
          END DO
          AINV(I,I) = 1.0D0
        END DO
        RETURN
      END IF

      DINV = 1.0D0 / DET

      AINV(1,1) = (A(2,2)*A(3,3) - A(2,3)*A(3,2)) * DINV
      AINV(1,2) = (A(1,3)*A(3,2) - A(1,2)*A(3,3)) * DINV
      AINV(1,3) = (A(1,2)*A(2,3) - A(1,3)*A(2,2)) * DINV
      AINV(2,1) = (A(2,3)*A(3,1) - A(2,1)*A(3,3)) * DINV
      AINV(2,2) = (A(1,1)*A(3,3) - A(1,3)*A(3,1)) * DINV
      AINV(2,3) = (A(1,3)*A(2,1) - A(1,1)*A(2,3)) * DINV
      AINV(3,1) = (A(2,1)*A(3,2) - A(2,2)*A(3,1)) * DINV
      AINV(3,2) = (A(1,2)*A(3,1) - A(1,1)*A(3,2)) * DINV
      AINV(3,3) = (A(1,1)*A(2,2) - A(1,2)*A(2,1)) * DINV

      RETURN
      END
