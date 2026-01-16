import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os # Needed for saving files
from datetime import datetime # --- NEW IMPORT ---

# =============================================================================
# ## 1. 変数定義 (Variable Definitions)
# =============================================================================
print("Setting up parameters...")

# --- 相互作用係数 (a_ij) ---
a11 = 1.0
a12 = 5.0
a13 = 5.0
a14 = 5.0
a22 = 1.0
a23 = 3.0
a24 = 3.0
a33 = 1.0
a34 = 2.0
a44 = 1.0
A = np.array([
    [a11, a12, a13, a14],
    [a12, a22, a23, a24], # Assuming a21 = a12
    [a13, a23, a33, a34], # Assuming a31 = a13
    [a14, a24, a34, a44]  # Assuming a41 = a14
])

# --- 抗生物質感受性 (b_ii) ---
b11 = 0.4
b22 = 0.3
b33 = 0.2
b44 = 0.1
b_diag = np.array([b11, b22, b33, b44])

# --- 粘性 (Viscosities) ---
Eta1 = 0.8
Eta2 = 1.0
Eta3 = 1.5
Eta4 = 2.0
Eta_vec = np.array([Eta1, Eta2, Eta3, Eta4])
Eta_phi_vec = Eta_vec.copy()

# --- ペナルティ係数 (Penalty Parameter) ---
Kp1 = 1e-4

# --- ソルバーのパラメータ (Solver Parameters) ---
dt = 1e-4             # タイムステップ
maxtimestep = 1500    # ステップの総数
eps = 1e-3            # ニュートン法の許容誤差
one = 0.999           # (Mathematica 'one' variable)
tmax = maxtimestep * dt
tt = 0.0              # 現在の時間を初期化

# --- 栄養供給と抗生物質レベル (Nutrient & Antibiotic) ---
def c(t):
    return 50.0 + 50.0 * np.sin(500.0 * t)

def alpha(t):
    return 10.0

# =============================================================================
# ## 1.5 ファイル名とフォルダのセットアップ (Folder & Filename Setup)
# =============================================================================
#
# --- THIS IS THE NEW SECTION ---
#
# 1. ベースとなるファイル名を定義
base_filename = "AusgabeFall01-changing"

# 2. タイムスタンプを生成 (例: 20251112_160500)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 3. タイムスタンプ付きのフォルダ名を作成
folder_name = f"results_{timestamp}"

# 4. フォルダを作成
os.makedirs(folder_name, exist_ok=True)
print(f"Created results folder: {folder_name}")

# 5. すべてのファイルパスを、新しいフォルダとタイムスタンプを使うように更新
varDateiname = os.path.join(folder_name, f"{base_filename}_params_{timestamp}.txt")
dataDateiname = os.path.join(folder_name, f"{base_filename}_data_{timestamp}.dat")

# (プロットのファイル名もここで定義しておくとクリーンです)
plot_monitor1_name = os.path.join(folder_name, f"{base_filename}_monitor1_{timestamp}.png")
plot_monitor2_name = os.path.join(folder_name, f"{base_filename}_monitor2_{timestamp}.png")
plot_monitor3_name = os.path.join(folder_name, f"{base_filename}_monitor3_{timestamp}.png")
plot_iterations_name = os.path.join(folder_name, f"{base_filename}_iterations_{timestamp}.png")
plot_condition_name = os.path.join(folder_name, f"{base_filename}_conditioning_{timestamp}.png")

# =============================================================================
# ## 1.6 パラメータを保存 (Save Parameters)
# =============================================================================
# (このセクションは変更不要です。'varDateiname' を自動的に使います)
#
print(f"Saving parameters to {varDateiname}...")
ahat11 = a11/Eta1
ahat12 = a12/Eta1
ahat21 = a12/Eta2
ahat22 = a22/Eta2
# ... (all other ahat/a/b variables)

with open(varDateiname, "w") as f: # "w" = write (clobber old file)
    f.write(f"maxtimestep = {maxtimestep}\n")
    f.write(f"dt = {dt}\n")
    f.write(f"eps = {eps}\n")
    f.write(f"ahat11 = {ahat11}\n")
    f.write(f"ahat12 = {ahat12}\n")
    f.write(f"a11 = {a11}\n")
    f.write(f"a12 = {a12}\n")
    f.write(f"b11 = {b11}\n")
    f.write(f"b22 = {b22}\n")
    f.write(f"b33 = {b33}\n")
    f.write(f"b44 = {b44}\n")
    f.write(f"Eta1 = {Eta1}\n")
    f.write(f"Eta2 = {Eta2}\n")
    f.write(f"Eta3 = {Eta3}\n")
    f.write(f"Eta4 = {Eta4}\n")
    f.write(f"Kp1 = {Kp1}\n")
    f.write(f"c_function = 50 + 50 * Sin[500 t]\n")
    f.write(f"alpha_function = 10\n")

# =============================================================================
# ## 2. 初期条件とモニター (Initial Conditions & Monitors)
# =============================================================================
# (このセクションは変更ありません)
#
print("Initializing state vectors and monitors...")
Phi1i = 0.02
Phi2i = 0.02
Phi3i = 0.02
Phi4i = 0.02
Phi5i = 1.0 - (Phi1i + Phi2i + Phi3i + Phi4i)
g_prev = np.array([Phi1i, Phi2i, Phi3i, Phi4i, Phi5i, one, one, one, one, eps])
g_new_guess = np.array([eps, eps, eps, eps, one, one, one, one, one, 0.0])
monitor_phi1 = [Phi1i]
monitor_phi2 = [Phi2i]
monitor_phi3 = [Phi3i]
monitor_phi4 = [Phi4i]
monitor_phi0 = [Phi5i]
monitor_psi1 = [one]
monitor_psi2 = [one]
monitor_psi3 = [one]
monitor_psi4 = [one]
monitor_sum = [np.sum(g_prev[0:5])]
monitor_phipsi1 = [Phi1i * one]
monitor_phipsi2 = [Phi2i * one]
monitor_phipsi3 = [Phi3i * one]
monitor_phipsi4 = [Phi4i * one]
monitor_c = [c(0.0)]
monitor_alpha = [alpha(0.0)]
monitor_time = [0.0]
monitor_iterations = []
monitor_detK = []
monitor_conditioning = []

# =============================================================================
# ## 3. システム関数 (System Functions)
# =============================================================================
# (このセクションは変更ありません)

def compute_Q_vector(g_new, g_old, t, dt):
    """
    Mathematicaの「exp // Simplify」の出力を計算します。
    (*** psi のペナルティ項のバグを修正済み ***)
    """
    phi_new = g_new[0:4]
    phi0_new = g_new[4]
    psi_new = g_new[5:9]
    gamma_new = g_new[9]
    phi_old = g_old[0:4]
    phi0_old = g_old[4]
    psi_old = g_old[5:9]
    phidot = (phi_new - phi_old) / dt
    phi0dot = (phi0_new - phi0_old) / dt
    psidot = (psi_new - psi_old) / dt
    
    Q = np.zeros(10)
    CapitalPhi = phi_new * psi_new
    Interaction_dot_product = A @ CapitalPhi
    c_t_value = c(t)
    term1_phi = (Kp1 * (2.0 - 4.0 * phi_new)) / (np.power(phi_new - 1.0, 3) * np.power(phi_new, 3))
    term2_phi = (1.0 / Eta_vec) * (gamma_new + \
                 (Eta_phi_vec + Eta_vec * psi_new**2) * phidot + \
                 Eta_vec * phi_new * psi_new * psidot)
    term3_phi = (c_t_value / Eta_vec) * psi_new * Interaction_dot_product
    Q[0:4] = term1_phi + term2_phi - term3_phi
    Q[4] = gamma_new + \
           (Kp1 * (2.0 - 4.0 * phi0_new)) / (np.power(phi0_new - 1.0, 3) * np.power(phi0_new, 3)) + \
           phi0dot
    term1_psi = (-2.0 * Kp1) / (np.power(psi_new - 1.0, 2) * np.power(psi_new, 3)) - \
                (2.0 * Kp1) / (np.power(psi_new - 1.0, 3) * np.power(psi_new, 2))
    term2_psi = (b_diag * alpha(t) / Eta_vec) * psi_new
    term3_psi = phi_new * psi_new * phidot + phi_new**2 * psidot
    term4_psi = (c_t_value / Eta_vec) * phi_new * Interaction_dot_product
    Q[5:9] = term1_psi + term2_psi + term3_psi - term4_psi
    Q[9] = np.sum(phi_new) + phi0_new - 1.0
    return Q

def compute_Q_vector(g_new, g_old, t, dt):
    """
    Mathematicaの「exp // Simplify」の出力を計算します。
    (*** psi のペナルティ項のバグを修正済み ***)
    """
    
    # --- 1. 変数の展開 (Unpack variables) ---
    # 10個の要素を持つ g_new ベクトルを、分かりやすい名前に分解します。
    # Unpacks the 10-element g_new vector into human-readable names.
    phi_new = g_new[0:4]      # v1-v4 (phi 1-4 の現在ステップの値)
    phi0_new = g_new[4]       # v5    (phi 0 の現在ステップの値)
    psi_new = g_new[5:9]      # v6-v9 (psi 1-4 の現在ステップの値)
    gamma_new = g_new[9]      # v10   (gamma の現在ステップの値)

    # 1ステップ前（t-dt）の、すでに確定した値を分解します。
    # Unpacks the "previous" (t-dt) confirmed values.
    phi_old = g_old[0:4]
    phi0_old = g_old[4]
    psi_old = g_old[5:9]
    
    # --- 2. 時間微分の近似 (Approximate time derivatives) ---
    # "速度" (vd) を、有限差分法 (v_new - v_old) / dt で近似計算します。
    # Approximates the "velocity" (vd) using the finite difference method: (v_new - v_old) / dt
    phidot = (phi_new - phi_old) / dt          # vd1-vd4
    phi0dot = (phi0_new - phi0_old) / dt       # vd5
    psidot = (psi_new - psi_old) / dt          # vd6-vd9

    # --- 3. 10個の方程式（Qベクトル）の準備 ---
    # 最終的に返す10個の「エラー値」を格納する空のベクトルを準備します。
    # Prepares an empty vector to store the 10 "error values" we will return.
    Q = np.zeros(10)
    
    # --- 4. 共有項（中間計算）(Shared Terms) ---
    # これらの方程式は非常に複雑なので、何度も登場する部分を先に計算しておきます。
    # These equations are complex, so we pre-calculate parts that appear multiple times.
    
    # CapitalPhi (Φ) は、量(phi)と質(psi)を掛け合わせた「活動量」です。
    # CapitalPhi (Φ) is the "activity level", a product of quantity (phi) and quality (psi).
    CapitalPhi = phi_new * psi_new
    
    # A @ CapitalPhi は、A行列（競争ルール）と活動量(CapitalPhi)の行列積です。
    # This is the matrix product of the interaction rules (A) and the activity levels (CapitalPhi).
    Interaction_dot_product = A @ CapitalPhi
    
    # 現在時刻 t における「栄養(c)」の量を取得します。
    # Gets the current amount of nutrient (c) at time t.
    c_t_value = c(t)

    # --- 5. 方程式 1-4 (phi の方程式) の計算 ---
    # Q[0:4] = (ペナルティ項) + (散逸/制約項) - (成長/相互作用項)

    # (Kp1 * (2.0 - 4.0 * phi_new)) / ...
    # これは「ペナルティ項」です。phi が 0 や 1 に近づくのを防ぎます。
    # This is the "penalty term". It prevents phi from reaching 0 or 1.
    term1_phi = (Kp1 * (2.0 - 4.0 * phi_new)) / (np.power(phi_new - 1.0, 3) * np.power(phi_new, 3))
    
    # (1.0 / Eta_vec) * (gamma_new + ...
    # これは「散逸項」（摩擦/粘性）と「制約項」(gamma) を合わせたものです。
    # This combines the "dissipation term" (friction/viscosity) and the "constraint term" (gamma).
    term2_phi = (1.0 / Eta_vec) * (gamma_new + \
                 (Eta_phi_vec + Eta_vec * psi_new**2) * phidot + \
                 Eta_vec * phi_new * psi_new * psidot)
    
    # (c_t_value / Eta_vec) * psi_new * Interaction_dot_product
    # これが「成長・相互作用項」です。栄養(c_t_value)と競争(Interaction_dot_product)が成長にどう影響するかをモデル化します。
    # This is the "growth/interaction term". It models how nutrients (c_t_value) and competition (Interaction_dot_product) affect growth.
    term3_phi = (c_t_value / Eta_vec) * psi_new * Interaction_dot_product
    
    # 3つの項を足し合わせて、phi に関する最初の4つの方程式（Q[0]～Q[3]）が完成します。
    # The 3 terms are combined to create the first 4 equations (Q[0] to Q[3]) for phi.
    Q[0:4] = term1_phi + term2_phi - term3_phi
    
    # --- 6. 方程式 5 (phi0 の方程式) の計算 ---
    # Q[4] = (制約項) + (ペナルティ項) + (散逸項)
    
    # 空きスペース(phi0)に関する方程式です。phi1-4と似ていますが、相互作用項はありません。
    # This is the equation for the empty space (phi0). It's similar to phi1-4 but has no interaction term.
    Q[4] = gamma_new + \
           (Kp1 * (2.0 - 4.0 * phi0_new)) / (np.power(phi0_new - 1.0, 3) * np.power(phi0_new, 3)) + \
           phi0dot
    
    # --- 7. 方程式 6-9 (psi の方程式) の計算 ---
    # Q[5:9] = (ペナルティ項) + (抗生物質項) + (散逸項) - (成長/相互作用項)

    # (-2.0 * Kp1) / ...
    # これは「psi」のためのペナルティ項です。phi のペナルティとは式が異なります。
    # This is the penalty term for "psi". Note its form is different from the penalty for phi.
    term1_psi = (-2.0 * Kp1) / (np.power(psi_new - 1.0, 2) * np.power(psi_new, 3)) - \
                (2.0 * Kp1) / (np.power(psi_new - 1.0, 3) * np.power(psi_new, 2))
    
    # (b_diag * alpha(t) / Eta_vec) * psi_new
    # これが「抗生物質項」です。抗生物質(alpha)が psi（健康状態など）をどれだけ悪化させるかをモデル化します。
    # This is the "antibiotic term". It models how antibiotics (alpha) degrade psi (health, etc.).
    term2_psi = (b_diag * alpha(t) / Eta_vec) * psi_new
    
    # phi_new * psi_new * phidot + phi_new**2 * psidot
    # これは「psi」に関する散逸項です。
    # This is the dissipation term related to psi.
    term3_psi = phi_new * psi_new * phidot + phi_new**2 * psidot
    
    # (c_t_value / Eta_vec) * phi_new * Interaction_dot_product
    # これは psi にとっての「成長・相互作用項」です。
    # This is the "growth/interaction term" from psi's perspective.
    term4_psi = (c_t_value / Eta_vec) * phi_new * Interaction_dot_product
    
    # 4つの項を足し合わせて、psi に関する次の方程式（Q[5]～Q[8]）が完成します。
    # The 4 terms are combined to create the next 4 equations (Q[5] to Q[8]) for psi.
    Q[5:9] = term1_psi + term2_psi + term3_psi - term4_psi
    
    # --- 8. 方程式 10 (gamma の制約方程式) ---
    # Q[9] = (制約ルール)
    
    # これが10番目の方程式です。物理法則ではなく、「ルール」そのものです。
    # This is the 10th equation. It is not a physical law, but the constraint rule itself.
    # 「すべての phi (v1-v5) の合計は 1.0 でなければならない」というルールを表します。
    # It represents the rule: "The sum of all phi (v1-v5) must equal 1.0".
    Q[9] = np.sum(phi_new) + phi0_new - 1.0
    
    # --- 9. 最終的なQベクトルを返す ---
    # 10個のエラー値（すべてが 0 になるべき値）をソルバーに返します。
    # Returns the 10 error values (which should all be 0) to the solver.
    return Q

# ---------------------------------------------------------------------------

def compute_Jacobian_matrix(g_new, g_old, t, dt):
    """
    CForm K の出力を実装します。
    (*** これは、すべての転写バグが修正された、
           完全に新しいヤコビアン関数です ***)
    """
    v = g_new
    phi_new = g_new[0:4]
    phi0_new = g_new[4]
    psi_new = g_new[5:9]
    phidot = (phi_new - g_old[0:4]) / dt
    psidot = (psi_new - g_old[5:9]) / dt
    c_t_value = c(t)
    CapitalPhi = phi_new * psi_new
    Interaction_dot_product = A @ CapitalPhi
    K = np.zeros((10, 10))
    phi_p_deriv = (Kp1*(-4. + 8.*v[0:4]))/(np.power(v[0:4],3)*np.power(v[0:4]-1.,3)) - \
                    (Kp1*(2. - 4.*v[0:4]))*(3./(np.power(v[0:4],4)*np.power(v[0:4]-1.,3)) + 3./(np.power(v[0:4],3)*np.power(v[0:4]-1.,4)))
    phi0_p_deriv = (Kp1*(-4. + 8.*v[4]))/(np.power(v[4],3)*np.power(v[4]-1.,3)) - \
                     (Kp1*(2. - 4.*v[4]))*(3./(np.power(v[4],4)*np.power(v[4]-1.,3)) + 3./(np.power(v[4],3)*np.power(v[4]-1.,4)))
    psi_p_deriv = (4.0 * Kp1 * (3.0 - 5.0*v[5:9] + 5.0*v[5:9]**2)) / (np.power(v[5:9], 4) * np.power(v[5:9] - 1.0, 4))
    for i in range(4):
        for j in range(4):
            K[i, j] = (c_t_value / Eta_vec[i]) * psi_new[i] * (-A[i, j] * psi_new[j])
        K[i, i] = phi_p_deriv[i] + \
                  (1.0 / Eta_vec[i]) * ( (Eta_phi_vec[i] + Eta_vec[i] * psi_new[i]**2) / dt + Eta_vec[i] * psi_new[i] * psidot[i] ) - \
                  (c_t_value / Eta_vec[i]) * ( psi_new[i] * (Interaction_dot_product[i] + A[i, i] * psi_new[i]) )
        K[i, 4] = 0.0
        for j in range(4):
            K[i, j+5] = (c_t_value / Eta_vec[i]) * psi_new[i] * (-A[i, j] * phi_new[j])
        K[i, i+5] = (1.0 / Eta_vec[i]) * ( 2.0 * Eta_vec[i] * psi_new[i] * phidot[i] + Eta_vec[i] * phi_new[i] * psidot[i] + Eta_vec[i] * phi_new[i] * psi_new[i] / dt ) - \
                      (c_t_value / Eta_vec[i]) * ( (Interaction_dot_product[i] + A[i, i] * phi_new[i] * psi_new[i]) + psi_new[i] * (A[i, i] * phi_new[i]) )
        K[i, 9] = 1.0 / Eta_vec[i]
    K[4, 4] = phi0_p_deriv + 1.0/dt
    K[4, 9] = 1.0
    for i in range(4):
        k = i + 5
        for j in range(4):
            K[k, j] = - (c_t_value / Eta_vec[i]) * ( A[i, j] * psi_new[j] * (phi_new[i]) + (Interaction_dot_product[i]) * (1.0 if i == j else 0.0) )
        K[k, i] = (psi_new[i] * phidot[i] + psi_new[i] * phi_new[i] / dt + 2.0 * phi_new[i] * psidot[i]) - \
                    (c_t_value / Eta_vec[i]) * ( A[i, i] * psi_new[i] * (phi_new[i]) + (Interaction_dot_product[i]) + phi_new[i] * (A[i, i] * psi_new[i]) )
        K[k, 4] = 0.0
        for j in range(4):
            K[k, j+5] = - (c_t_value / Eta_vec[i]) * phi_new[i] * (A[i, j] * phi_new[j])
        K[k, i+5] = psi_p_deriv[i] + \
                      (b_diag[i] * alpha(t) / Eta_vec[i]) + \
                      (phi_new[i] * phidot[i] + phi_new[i]**2 / dt) - \
                      (c_t_value / Eta_vec[i]) * phi_new[i] * (A[i, i] * phi_new[i])
        K[k, 9] = 0.0
    K[9, 0] = 1.0
    K[9, 1] = 1.0
    K[9, 2] = 1.0
    K[9, 3] = 1.0
    K[9, 4] = 1.0
    return K

def compute_Jacobian_matrix(g_new, g_old, t, dt):
    """
    CForm K の出力を実装します。
    (*** これは、すべての転写バグが修正された、
           完全に新しいヤコビアン関数です ***)
    """
    
    # --- 1. 変数の展開 (Unpack variables) ---
    # Qベクトルと同様に、計算しやすいように変数に名前を付けます。
    # Just like in Q_vector, we assign names for easier calculation.
    v = g_new
    phi_new = g_new[0:4]      # v1-v4 (phi 1-4)
    phi0_new = g_new[4]       # v5    (phi 0)
    psi_new = g_new[5:9]      # v6-v9 (psi 1-4)
    
    # --- 2. 時間微分の計算 (Calculate time derivatives) ---
    # K行列の計算には「速度」の値そのものが必要なため、ここで計算します。
    # We need the "velocity" values themselves to calculate the K matrix.
    phidot = (phi_new - g_old[0:4]) / dt          # vd1-vd4
    psidot = (psi_new - g_old[5:9]) / dt          # vd6-vd9

    # --- 3. 共有項の計算 (Calculate Shared Terms) ---
    # Qベクトルと同様に、何度も使う中間計算を先に行います。
    # Just like in Q_vector, pre-calculate intermediate terms used many times.
    c_t_value = c(t)
    CapitalPhi = phi_new * psi_new
    Interaction_dot_product = A @ CapitalPhi
    
    # --- 4. K行列の準備 (Initialize K Matrix) ---
    # 10x10 の「空の（ゼロで埋められた）」行列を準備します。
    # Prepares an empty (zero-filled) 10x10 matrix.
    K = np.zeros((10, 10))

    # --- 5. 共有の微分項の計算 (Calculate Shared Derivative Terms) ---
    # K行列の中には、同じ形の「微分（変化率）」が何度も登場します。
    # 特にペナルティ項の微分は複雑なので、ここで先に計算しておきます。
    # This saves computation time by pre-calculating complex derivatives.

    # phi のペナルティ項 (term1_phi) を、phi (v[0:4]) で微分した値。
    # The derivative of phi's penalty term (term1_phi) with respect to phi (v[0:4]).
    phi_p_deriv = (Kp1*(-4. + 8.*v[0:4]))/(np.power(v[0:4],3)*np.power(v[0:4]-1.,3)) - \
                    (Kp1*(2. - 4.*v[0:4]))*(3./(np.power(v[0:4],4)*np.power(v[0:4]-1.,3)) + 3./(np.power(v[0:4],3)*np.power(v[0:4]-1.,4)))
    
    # phi0 のペナルティ項 (Q[4]) を、phi0 (v[4]) で微分した値。
    # The derivative of phi0's penalty term (Q[4]) with respect to phi0 (v[4]).
    phi0_p_deriv = (Kp1*(-4. + 8.*v[4]))/(np.power(v[4],3)*np.power(v[4]-1.,3)) - \
                     (Kp1*(2. - 4.*v[4]))*(3./(np.power(v[4],4)*np.power(v[4]-1.,3)) + 3./(np.power(v[4],3)*np.power(v[4]-1.,4)))
    
    # psi のペナルティ項 (term1_psi) を、psi (v[5:9]) で微分した値。
    # The derivative of psi's penalty term (term1_psi) with respect to psi (v[5:9]).
    psi_p_deriv = (4.0 * Kp1 * (3.0 - 5.0*v[5:9] + 5.0*v[5:9]**2)) / (np.power(v[5:9], 4) * np.power(v[5:9] - 1.0, 4))
    
    
    # --- 6. 行列の要素を代入 (Populate Matrix Elements) ---
    # ここから、K行列の100個のマスを埋めていきます。
    # Now we fill in the 100 elements of the K matrix.
    # K[i, j] は「方程式 Q[i] を、変数 v[j] で微分した値」です。
    # K[i, j] is the derivative of Equation Q[i] with respect to variable v[j].

    # --- 行 0-3 (Rows 0-3): [d(Q_phi) / d(v_j)] ---
    # Q[0]～Q[3] (phi の方程式) を、各変数 v[j] で微分します。
    # Derivatives of Q[0]-Q[3] (the phi equations) with respect to each variable v[j].
    for i in range(4): # i = 0 から 3 (phi1 から phi4 の方程式)
        
        # --- K[i, 0:4] (phi方程式 を phi変数 で微分) ---
        for j in range(4): # j = 0 から 3 (phi1 から phi4 の変数)
            # (対角成分以外: K[i, j] where i != j)
            # phi_i の方程式は、phi_j (i!=j) が変わると、相互作用項(term3_phi)だけが変化します。
            # The equation for phi_i only changes via the interaction term (term3_phi) when phi_j (i!=j) changes.
            K[i, j] = (c_t_value / Eta_vec[i]) * psi_new[i] * (-A[i, j] * psi_new[j])
        
        # (対角成分: K[i, i])
        # phi_i の方程式を phi_i 自身で微分します。
        # (ペナルティ項、散逸項、相互作用項のすべてが変化するため、最も複雑な項の一つ)
        # Derivative of phi_i's equation with respect to itself. One of the most complex terms.
        K[i, i] = phi_p_deriv[i] + \
                  (1.0 / Eta_vec[i]) * ( (Eta_phi_vec[i] + Eta_vec[i] * psi_new[i]**2) / dt + Eta_vec[i] * psi_new[i] * psidot[i] ) - \
                  (c_t_value / Eta_vec[i]) * ( psi_new[i] * (Interaction_dot_product[i] + A[i, i] * psi_new[i]) )
        
        # --- K[i, 4] (phi方程式 を phi0 で微分) ---
        # phi の方程式は phi0 に依存しないので、微分は 0 です。
        # The phi equations do not depend on phi0, so the derivative is 0.
        K[i, 4] = 0.0
        
        # --- K[i, 5:9] (phi方程式 を psi変数 で微分) ---
        for j in range(4): # j = 0 から 3 (psi1 から psi4 の変数に対応)
            # (対角成分以外: K[i, j+5] where i != j)
            # phi_i の方程式は、psi_j (i!=j) が変わると、相互作用項(term3_phi)だけが変化します。
            # The phi_i equation only changes via the interaction term when psi_j (i!=j) changes.
            K[i, j+5] = (c_t_value / Eta_vec[i]) * psi_new[i] * (-A[i, j] * phi_new[j])
        
        # (対角成分: K[i, i+5])
        # phi_i の方程式を psi_i で微分します。(散逸項と相互作用項が変化します)
        # Derivative of phi_i's equation with respect to psi_i. (Dissipation and interaction terms change).
        K[i, i+5] = (1.0 / Eta_vec[i]) * ( 2.0 * Eta_vec[i] * psi_new[i] * phidot[i] + Eta_vec[i] * phi_new[i] * psidot[i] + Eta_vec[i] * phi_new[i] * psi_new[i] / dt ) - \
                      (c_t_value / Eta_vec[i]) * ( (Interaction_dot_product[i] + A[i, i] * phi_new[i] * psi_new[i]) + psi_new[i] * (A[i, i] * phi_new[i]) )

        # --- K[i, 9] (phi方程式 を gamma で微分) ---
        # Q[i] の term2_phi には (1.0 / Eta_vec[i]) * gamma_new が含まれるため、gamma で微分するとその係数が残ります。
        # The term2_phi in Q[i] includes (1.0 / Eta_vec[i]) * gamma_new. Differentiating by gamma leaves this coefficient.
        K[i, 9] = 1.0 / Eta_vec[i]

    # --- 行 4 (Row 4): [d(Q_phi0) / d(v_j)] ---
    # Q[4] (phi0 の方程式) を、各変数 v[j] で微分します。
    # Derivatives of Q[4] (the phi0 equation) with respect to each variable v[j].
    
    # Q[4] は v[4] (phi0) と v[9] (gamma) にしか依存しません。
    # Q[4] only depends on v[4] (phi0) and v[9] (gamma).
    K[4, 4] = phi0_p_deriv + 1.0/dt # (ペナルティ項の微分 + 散逸項(phi0dot)の微分)
    K[4, 9] = 1.0                   # (gamma_new の微分)
    # (他の K[4, 0:4], K[4, 5:9] はすべて 0 のままです)
    # (All other elements K[4, 0:4] and K[4, 5:9] remain 0)

    # --- 行 5-8 (Rows 5-8): [d(Q_psi) / d(v_j)] ---
    # Q[5]～Q[8] (psi の方程式) を、各変数 v[j] で微分します。
    # Derivatives of Q[5]-Q[8] (the psi equations) with respect to each variable v[j].
    for i in range(4): # i = 0 から 3 (psi1 から psi4 の方程式)
        k = i + 5 # k = 5 から 8 (K行列の行インデックス)
        
        # --- K[k, 0:4] (psi方程式 を phi変数 で微分) ---
        for j in range(4): # j = 0 から 3 (phi1 から phi4 の変数)
            # (対角成分以外: K[k, j] where i != j)
            # psi_i の方程式は、phi_j (i!=j) が変わると、相互作用項(term4_psi)だけが変化します。
            # The psi_i equation only changes via the interaction term (term4_psi) when phi_j (i!=j) changes.
            K[k, j] = - (c_t_value / Eta_vec[i]) * ( A[i, j] * psi_new[j] * (phi_new[i]) + (Interaction_dot_product[i]) * (1.0 if i == j else 0.0) )
        
        # (対角成分: K[k, i])
        # psi_i の方程式を phi_i で微分します。(散逸項と相互作用項が変化します)
        # Derivative of psi_i's equation with respect to phi_i. (Dissipation and interaction terms change).
        K[k, i] = (psi_new[i] * phidot[i] + psi_new[i] * phi_new[i] / dt + 2.0 * phi_new[i] * psidot[i]) - \
                    (c_t_value / Eta_vec[i]) * ( A[i, i] * psi_new[i] * (phi_new[i]) + (Interaction_dot_product[i]) + phi_new[i] * (A[i, i] * psi_new[i]) )
        
        # --- K[k, 4] (psi方程式 を phi0 で微分) ---
        # psi の方程式は phi0 に依存しないので、微分は 0 です。
        # The psi equations do not depend on phi0, so the derivative is 0.
        K[k, 4] = 0.0
        
        # --- K[k, 5:9] (psi方程式 を psi変数 で微分) ---
        for j in range(4): # j = 0 から 3 (psi1 から psi4 の変数に対応)
            # (対角成分以外: K[k, j+5] where i != j)
            # psi_i の方程式は、psi_j (i!=j) が変わると、相互作用項(term4_psi)だけが変化します。
            # The psi_i equation only changes via the interaction term when psi_j (i!=j) changes.
            K[k, j+5] = - (c_t_value / Eta_vec[i]) * phi_new[i] * (A[i, j] * phi_new[j])
        
        # (対角成分: K[k, i+5])
        # psi_i の方程式を psi_i 自身で微分します。
        # (ペナルティ、抗生物質、散逸、相互作用のすべての項が変化します)
        # Derivative of psi_i's equation with respect to itself. (All 4 terms change).
        K[k, i+5] = psi_p_deriv[i] + \
                      (b_diag[i] * alpha(t) / Eta_vec[i]) + \
                      (phi_new[i] * phidot[i] + phi_new[i]**2 / dt) - \
                      (c_t_value / Eta_vec[i]) * phi_new[i] * (A[i, i] * phi_new[i])
        
        # --- K[k, 9] (psi方程式 を gamma で微分) ---
        # psi の方程式は gamma に依存しないので、微分は 0 です。
        # The psi equations do not depend on gamma, so the derivative is 0.
        K[k, 9] = 0.0

    # --- 行 9 (Row 9): [d(Q_gamma) / d(v_j)] ---
    # Q[9] (制約方程式) を、各変数 v[j] で微分します。
    # Derivatives of Q[9] (the constraint equation) with respect to each variable v[j].
    # Q[9] = v[0] + v[1] + v[2] + v[3] + v[4] - 1.0
    K[9, 0] = 1.0 # d(Q[9]) / dv[0] (phi1)
    K[9, 1] = 1.0 # d(Q[9]) / dv[1] (phi2)
    K[9, 2] = 1.0 # d(Q[9]) / dv[2] (phi3)
    K[9, 3] = 1.0 # d(Q[9]) / dv[3] (phi4)
    K[9, 4] = 1.0 # d(Q[9]) / dv[4] (phi0)
    # (他の K[9, 5:10] はすべて 0 のままです)
    # (All other elements K[9, 5:10] remain 0)
    
    # --- 7. 完成した地図（K行列）を返す ---
    # Returns the completed "map" (K matrix) to the solver.
    return K

# =============================================================================
# ## 4. ニュートン・ラプソン ソルバー (Newton-Raphson Solver)
# =============================================================================
# (このセクションは変更ありません)
#
print("Starting simulation...")
start_time = time.time()
np.seterr(divide='ignore', invalid='ignore')
K_matrix = np.zeros((10,10))

for step in range(maxtimestep):
    tt = (step + 1) * dt
    newton_iter = 0
    max_error = 1.0
    g_new = g_prev.copy()

    while (max_error > eps) and (newton_iter < 100):
        Q = compute_Q_vector(g_new, g_prev, tt, dt)
        K_matrix = compute_Jacobian_matrix(g_new, g_prev, tt, dt)

        if np.isnan(Q).any() or np.isnan(K_matrix).any():
            print(f"Error: NaN detected in Q or K at t={tt}")
            max_error = -2
            break

        try:
            dg = np.linalg.solve(K_matrix, -Q)
        except np.linalg.LinAlgError:
            print(f"Error: Jacobian matrix is singular at t={tt}")
            max_error = -1
            break

        g_new = g_new + dg
        max_error = np.max(np.abs(Q))
        newton_iter += 1

    if max_error == -1 or max_error == -2:
        print("Simulation stopped due to fatal error.")
        break
    # (We removed the warning for newton_iter >= 100)

    # --- モニターを更新 (Update Monitors) ---
    monitor_time.append(tt)
    monitor_phi1.append(g_new[0])
    monitor_phi2.append(g_new[1])
    monitor_phi3.append(g_new[2])
    monitor_phi4.append(g_new[3])
    monitor_phi0.append(g_new[4])
    monitor_psi1.append(g_new[5])
    monitor_psi2.append(g_new[6])
    monitor_psi3.append(g_new[7])
    monitor_psi4.append(g_new[8])
    monitor_sum.append(np.sum(g_new[0:5]))
    monitor_phipsi1.append(g_new[0] * g_new[5])
    monitor_phipsi2.append(g_new[1] * g_new[6])
    monitor_phipsi3.append(g_new[2] * g_new[7])
    monitor_phipsi4.append(g_new[3] * g_new[8])
    monitor_c.append(c(tt))
    monitor_alpha.append(alpha(tt))
    monitor_iterations.append(newton_iter)
    monitor_detK.append(np.linalg.det(K_matrix))
    monitor_conditioning.append(np.linalg.cond(K_matrix))

    g_prev = g_new.copy()

    if (step + 1) % 1000 == 0:
        print(f"Step {step+1}/{maxtimestep} (t = {tt:.4f}), Iterations: {newton_iter}, Max Error: {max_error:.2e}")

end_time = time.time()
print(f"Simulation finished in {end_time - start_time:.2f} seconds.")

# =============================================================================
# ## 4. ニュートン・ラプソン ソルバー (Newton-Raphson Solver)
# =============================================================================
#
print("Starting simulation...") # シミュレーション開始を通知
start_time = time.time() # 開始時刻を記録

# (ゼロ除算や無効な計算（例: inf * 0）に関するNumpyの警告を非表示に)
np.seterr(divide='ignore', invalid='ignore') 
# K行列（ヤコビアン）を格納する変数をあらかじめ初期化
K_matrix = np.zeros((10,10)) 

# --- 外側ループ (時間ループ) ---
# maxtimestep の回数（例: 1500回）だけ、時間を進めるループ
for step in range(maxtimestep):
    
    # --- 4a. 時間の更新と推測値の準備 ---
    
    # tt (現在時刻) を dt (0.0001秒) だけ進める
    tt = (step + 1) * dt 
    # newton法の試行回数カウンターをリセット
    newton_iter = 0 
    # newton法のエラーを「1.0」（epsより大きい値）にリセット
    max_error = 1.0 
    # 「現在の推測値(g_new)」として、ひとまず「前のステップの答え(g_prev)」をコピー
    g_new = g_prev.copy() 

    # --- 内側ループ (ニュートン法) ---
    # 「エラー(max_error)が目標値(eps)より大きい」かつ「試行回数(newton_iter)が100回未満」の間、実行
    while (max_error > eps) and (newton_iter < 100):
        
        # 1. 「現在の方程式のエラー(Q)」を計算
        Q = compute_Q_vector(g_new, g_prev, tt, dt) 
        # 2. 「現在の地図(K_matrix)」を計算
        K_matrix = compute_Jacobian_matrix(g_new, g_prev, tt, dt) 

        # 3. 安全装置(1): 計算結果が NaN (計算不能) になったらループを脱出
        if np.isnan(Q).any() or np.isnan(K_matrix).any():
            print(f"Error: NaN detected in Q or K at t={tt}")
            max_error = -2 # エラーコード -2 (NaN) を設定
            break # 内側ループを強制終了

        # 4. 「最適な修正量(dg)」を計算
        try:
            # 線形方程式 K * dg = -Q を解いて、dg を見つける
            dg = np.linalg.solve(K_matrix, -Q) 
        # 5. 安全装置(2): K行列が特異行列(singular)で解けない場合
        except np.linalg.LinAlgError: 
            print(f"Error: Jacobian matrix is singular at t={tt}")
            max_error = -1 # エラーコード -1 (Singular) を設定
            break # 内側ループを強制終了

        # 6. 推測値の更新: 「推測値」に「修正量」を足して、より良い推測値にする
        g_new = g_new + dg 
        # 7. エラーの再評価: 新しい推測値での最大エラーを計算
        max_error = np.max(np.abs(Q)) 
        # 8. 試行回数をカウントアップ
        newton_iter += 1 

    # --- 4b. 内側ループ終了後の処理 ---
    
    # もし安全装置が作動していたら、シミュレーション全体（外側ループ）を停止
    if max_error == -1 or max_error == -2: 
        print("Simulation stopped due to fatal error.")
        break # 外側ループを強制終了
    
    # (試行回数が100回に達してループが終了した場合の警告は、コメントアウトされています)
    # (We removed the warning for newton_iter >= 100)

    # --- 4c. モニター（実験ノート）への記録 ---
    # whileループが見つけた「確定した答え(g_new)」を、リストに1行ずつ追加
    monitor_time.append(tt)
    monitor_phi1.append(g_new[0])
    monitor_phi2.append(g_new[1])
    monitor_phi3.append(g_new[2])
    monitor_phi4.append(g_new[3])
    monitor_phi0.append(g_new[4])
    monitor_psi1.append(g_new[5])
    monitor_psi2.append(g_new[6])
    monitor_psi3.append(g_new[7])
    monitor_psi4.append(g_new[8])
    monitor_sum.append(np.sum(g_new[0:5])) # phiの合計値も記録
    monitor_phipsi1.append(g_new[0] * g_new[5]) # (phi*psi)も記録
    monitor_phipsi2.append(g_new[1] * g_new[6])
    monitor_phipsi3.append(g_new[2] * g_new[7])
    monitor_phipsi4.append(g_new[3] * g_new[8])
    monitor_c.append(c(tt)) # その時刻の栄養(c)の値も記録
    monitor_alpha.append(alpha(tt)) # その時刻の抗生物質(alpha)の値も記録
    monitor_iterations.append(newton_iter) # このステップにかかった試行回数も記録
    monitor_detK.append(np.linalg.det(K_matrix)) # K行列の行列式を記録
    monitor_conditioning.append(np.linalg.cond(K_matrix)) # K行列の条件数を記録

    # --- 4d. 次のステップへの準備 ---
    # 見つけた「確定した答え(g_new)」を、「次のステップ」のための「1つ前の答え(g_prev)」として保存
    g_prev = g_new.copy() 

    # --- 4e. 進捗報告 ---
    # 1000ステップごとに、現在の状況を画面に出力
    if (step + 1) % 1000 == 0: 
        print(f"Step {step+1}/{maxtimestep} (t = {tt:.4f}), Iterations: {newton_iter}, Max Error: {max_error:.2e}")

# --- 4f. 終了処理 ---
# 外側ループが完了したら、かかった合計時間を計算して表示
end_time = time.time() 
print(f"Simulation finished in {end_time - start_time:.2f} seconds.")

# =============================================================================
# ## 5. 出力とプロット (Output and Plotting)
# =============================================================================
#
# --- 提案された3つの新しいグラフをすべて含んでいます ---
#
print("Plotting results...")

# --- Plot monitor1 (phi, psi, sum) ---
plt.figure(figsize=(10, 6))
plt.plot(monitor_time, monitor_phi1, label='phi_1', color='red', linestyle='-')
plt.plot(monitor_time, monitor_phi2, label='phi_2', color='blue', linestyle='-')
plt.plot(monitor_time, monitor_phi3, label='phi_3', color='yellow', linestyle='-')
plt.plot(monitor_time, monitor_phi4, label='phi_4', color='green', linestyle='-')
plt.plot(monitor_time, monitor_phi0, label='phi_0 (Empty)', color='black', linestyle='-')
plt.plot(monitor_time, monitor_psi1, label='psi_1', color='red', linestyle='--')
plt.plot(monitor_time, monitor_psi2, label='psi_2', color='blue', linestyle='--')
plt.plot(monitor_time, monitor_psi3, label='psi_3', color='yellow', linestyle='--')
plt.plot(monitor_time, monitor_psi4, label='psi_4', color='green', linestyle='--')
plt.plot(monitor_time, monitor_sum, label='Sum(phis)', color='orange', linestyle='-')
plt.title(f'All Variables (monitor1) - {timestamp}')
plt.xlabel('Time (t)')
plt.ylabel('Value')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_monitor1_name, bbox_inches='tight')
print(f"Saved plot to {plot_monitor1_name}")
plt.show()

# --- Plot monitor2 (phi*psi) ---
plt.figure(figsize=(10, 6))
plt.plot(monitor_time, monitor_phipsi1, label='phi1*psi1', color='red')
plt.plot(monitor_time, monitor_phipsi2, label='phi2*psi2', color='blue')
plt.plot(monitor_time, monitor_phipsi3, label='phi3*psi3', color='yellow')
plt.plot(monitor_time, monitor_phipsi4, label='phi4*psi4', color='green')
plt.title(f'Interaction Variables (monitor2) - {timestamp}')
plt.xlabel('Time (t)')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.savefig(plot_monitor2_name)
print(f"Saved plot to {plot_monitor2_name}")
plt.show()

# --- Plot monitor3 (c, alpha) ---
plt.figure(figsize=(10, 4))
plt.plot(monitor_time, monitor_c, label='c (Nutrient)', color='green')
plt.plot(monitor_time, monitor_alpha, label='alpha (Antibiotic)', color='red')
plt.title(f'Inputs (monitor3) - {timestamp}')
plt.xlabel('Time (t)')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.savefig(plot_monitor3_name)
print(f"Saved plot to {plot_monitor3_name}")
plt.show()

# --- Plot cc (Iterations) ---
plt.figure(figsize=(10, 4))
plt.plot(monitor_iterations)
plt.title(f'Newton-Raphson Iterations per Time Step - {timestamp}')
plt.xlabel('Time Step')
plt.ylabel('Iterations')
plt.grid(True)
plt.savefig(plot_iterations_name)
print(f"Saved plot to {plot_iterations_name}")
plt.show()

# --- Plot myConditioning (Condition Number) ---
plt.figure(figsize=(10, 4))
plt.plot(monitor_conditioning)
plt.yscale('log')
plt.title(f'Jacobian Condition Number - {timestamp}')
plt.xlabel('Time Step')
plt.ylabel('Condition Number (Log Scale)')
plt.grid(True)
plt.savefig(plot_condition_name)
print(f"Saved plot to {plot_condition_name}")
plt.show()

# =============================================================================
# ## 5.5 新しいプロット (New Suggested Plots)
# =============================================================================

# --- (セクション4で計算されたリストを使用) ---
# 1. 総バイオマス (Total Biomass)
monitor_total_biomass = np.array(monitor_phi1) + np.array(monitor_phi2) + np.array(monitor_phi3) + np.array(monitor_phi4)
# (新しいファイル名を定義)
plot_total_biomass_name = os.path.join(folder_name, f"{base_filename}_total_biomass_{timestamp}.png")

plt.figure(figsize=(10, 4))
plt.plot(monitor_time, monitor_total_biomass, label='Total Biomass (phi1+2+3+4)', color='purple')
plt.title(f'Total Biomass Over Time - {timestamp}')
plt.xlabel('Time (t)')
plt.ylabel('Total Volume Fraction')
plt.legend()
plt.grid(True)
plt.savefig(plot_total_biomass_name)
print(f"Saved plot to {plot_total_biomass_name}")
plt.show()


# 2. 相対存在比 (Relative Abundance Stack Plot)
# (新しいファイル名を定義)
plot_relative_abundance_name = os.path.join(folder_name, f"{base_filename}_relative_abundance_{timestamp}.png")

phi1_data = np.array(monitor_phi1)
phi2_data = np.array(monitor_phi2)
phi3_data = np.array(monitor_phi3)
phi4_data = np.array(monitor_phi4)
total_biomass_data = monitor_total_biomass # (上で計算済み)

safe_total = np.where(total_biomass_data == 0, 1e-10, total_biomass_data)
rel_phi1 = phi1_data / safe_total
rel_phi2 = phi2_data / safe_total
rel_phi3 = phi3_data / safe_total
rel_phi4 = phi4_data / safe_total

plt.figure(figsize=(10, 6))
plt.stackplot(monitor_time, rel_phi1, rel_phi2, rel_phi3, rel_phi4, 
              labels=['Species 1 Share', 'Species 2 Share', 'Species 3 Share', 'Species 4 Share'],
              colors=['red', 'blue', 'yellow', 'green'])
plt.title(f'Relative Species Abundance (Market Share) - {timestamp}')
plt.xlabel('Time (t)')
plt.ylabel('Relative Abundance (Share of Total Biomass)')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_relative_abundance_name, bbox_inches='tight')
print(f"Saved plot to {plot_relative_abundance_name}")
plt.show()


# 3. 位相図 (Phase Plot)
# (新しいファイル名を定義)
plot_phase_plot_name = os.path.join(folder_name, f"{base_filename}_phase_plot_{timestamp}.png")

plt.figure(figsize=(6, 6))
plt.plot(monitor_c, monitor_total_biomass, label='System Trajectory')
plt.plot(monitor_c[0], monitor_total_biomass[0], 'go', label='Start') # 'go' = green circle
plt.plot(monitor_c[-1], monitor_total_biomass[-1], 'rs', label='End') # 'rs' = red square
plt.title(f'Phase Plot: Nutrient vs. Total Biomass - {timestamp}')
plt.xlabel('Nutrient Level (c(t))')
plt.ylabel('Total Biomass')
plt.legend()
plt.grid(True)
plt.savefig(plot_phase_plot_name)
print(f"Saved plot to {plot_phase_plot_name}")
plt.show()


# =============================================================================
# ## 6. 最終データのエクスポート (Final Data Export)
# =============================================================================
#
print(f"Exporting all data to {dataDateiname}...")

timestep_col = np.arange(len(monitor_time))
data_to_export = {
    "timestep": timestep_col,
    "phi1": monitor_phi1,
    "phi2": monitor_phi2,
    "phi3": monitor_phi3,
    "phi4": monitor_phi4,
    "phi0": monitor_phi0,
    "psi1": monitor_psi1,
    "psi2": monitor_psi2,
    "psi3": monitor_psi3,
    "psi4": monitor_psi4,
    "Sum(phis)": monitor_sum,
    "phi1psi1": monitor_phipsi1,
    "phi2psi2": monitor_phipsi2,
    "phi3psi3": monitor_phipsi3,
    "phi4psi4": monitor_phipsi4,
    "c": monitor_c,
    "alpha": monitor_alpha,
    # (新しい列をデータファイルにも追加)
    "TotalBiomass": monitor_total_biomass 
}

df = pd.DataFrame(data_to_export)
df.to_csv(dataDateiname, sep="\t", index=False)

print(f"Done. All files saved to folder: {folder_name}")