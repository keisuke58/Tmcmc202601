# Hamilton原理（理論）から今回実装（FOM/ROM）が妥当と言える範囲の「数学的検証ノート」

目的：`hamiltonian.pdf`（拡張Hamilton原理の一般論）と `biofilm_simulation.pdf`（biofilmモデルの具体化）を踏まえて、**今回の実装（特に `improved1207_paper_jit.py` のFOM）** が「論文で導出された支配方程式」を正しく解いているかを、数式レベルで対応付けて確認する。

このノートは「理論→強形式→離散化→コード」の順で、**証明できるところ**と**限界**（離散化・安定化処理）を明確にします。

---

## 1. 理論（biofilm_simulation.pdf）で定義されている要素

論文では内部変数を

- 体積分率：\(\phi_0,\phi_1,\dots,\phi_n\)（各 \(\in(0,1)\)）
- 生存率：\(\psi_1,\dots,\psi_n\)（各 \(\in(0,1)\)）
- 生菌量：\(\bar\phi_i=\phi_i\psi_i\)

として、**体積制約（holonomic constraint）**

\[
f(\phi)=\sum_{l=0}^{n}\phi_l-1=0
\]

をラグランジュ乗数 \(\gamma\) で課します。

自由エネルギー密度（isothermal/quasi-staticの材料点モデル）：

\[
\Psi(\phi,\psi)= -\frac12 c^\* \bar\phi^\top A\,\bar\phi + \frac12 \alpha^\* \psi^\top B\,\psi
\]

散逸ポテンシャル（速度依存＝rate-dependent）：

\[
\Delta_s(\dot{\bar\phi},\dot\phi)=\frac12 \dot{\bar\phi}^\top \eta\,\dot{\bar\phi} + \frac12 \dot\phi^\top \eta\,\dot\phi
\]

ここで \(\eta\) は対角（\(\eta_i>0\)）。

この構成は、`hamiltonian.pdf` の「拡張Hamilton原理（非保存・散逸を含む）を汎用枠組みとして、制約はラグランジュ項で入れ、散逸は散逸ポテンシャルで入れる」という一般論の **具体例** になっています。

---

## 2. 強形式（論文の(16)–(18)）がどう出るか：計算の骨格（証明スケッチ）

「拡張Hamilton機能 \(H\) の停留（変分ゼロ）」は、ここでは **内部変数の進化**を与える形になります（運動方程式というより、Onsager型の進化則に近い）。

重要なのは、以下の3点が“変分計算で必然的に出る”ことです。

- **(a) エネルギー項の寄与**：\(\partial\Psi/\partial \phi_i\) と \(\partial\Psi/\partial \psi_i\)
- **(b) 散逸項の寄与**：\(\partial \Delta_s/\partial \dot\phi_i\) と \(\partial \Delta_s/\partial \dot\psi_i\)（ただし \(\dot{\bar\phi}_i=\dot\phi_i\psi_i+\phi_i\dot\psi_i\) なので連鎖律でクロス項が出る）
- **(c) 制約の寄与**：\(\gamma\,\partial f/\partial \phi_i=\gamma\)、\(\partial f/\partial \psi_i=0\)（制約は \(\phi\) のみの関数）

### 2.1 エネルギー勾配（\(\Psi\) 由来）

\[
\frac{\partial\Psi}{\partial \bar\phi}= -c^\* A\bar\phi
\]

\(\bar\phi_i=\phi_i\psi_i\) より

\[
\frac{\partial\Psi}{\partial \phi_i}=\frac{\partial\Psi}{\partial \bar\phi_i}\frac{\partial\bar\phi_i}{\partial\phi_i}
=(-c^\*(A\bar\phi)_i)\,\psi_i
\]

\[
\frac{\partial\Psi}{\partial \psi_i}=\frac{\partial\Psi}{\partial \bar\phi_i}\frac{\partial\bar\phi_i}{\partial\psi_i}+\frac{\partial}{\partial\psi_i}\Big(\frac12\alpha^\*\psi^\top B\psi\Big)
=(-c^\*(A\bar\phi)_i)\,\phi_i+\alpha^\* b_i \psi_i
\]

### 2.2 散逸勾配（\(\Delta_s\) 由来）

散逸の第1項は \(\dot{\bar\phi}_i=\dot\phi_i\psi_i+\phi_i\dot\psi_i\) を通じて

\[
\frac{\partial}{\partial \dot\phi_i}\Big(\frac12 \eta_i \dot{\bar\phi}_i^2\Big)=\eta_i \dot{\bar\phi}_i\,\psi_i
=\eta_i(\dot\phi_i\psi_i^2+\phi_i\psi_i\dot\psi_i)
=\eta_i(\dot\phi_i\psi_i^2+\bar\phi_i\dot\psi_i)
\]

\[
\frac{\partial}{\partial \dot\psi_i}\Big(\frac12 \eta_i \dot{\bar\phi}_i^2\Big)=\eta_i \dot{\bar\phi}_i\,\phi_i
=\eta_i(\dot\psi_i\phi_i^2+\bar\phi_i\dot\phi_i)
\]

散逸の第2項 \(\frac12\dot\phi^\top\eta\dot\phi\) からさらに \(\eta_i\dot\phi_i\) が加わり、結果として

- \(\phi\)-方程式には \(\eta_i(\dot\phi_i\psi_i^2+\bar\phi_i\dot\psi_i+\dot\phi_i)\)
- \(\psi\)-方程式には \(\eta_i(\dot\psi_i\phi_i^2+\bar\phi_i\dot\phi_i)\)

が現れます。

### 2.3 制約（\(\gamma\)）の寄与

\[
f(\phi)=\sum_l\phi_l-1
\Rightarrow \frac{\partial f}{\partial \phi_i}=1,\quad \frac{\partial f}{\partial\psi_i}=0
\]

よって、\(\phi\) 側に \(\gamma\) が入り、\(\psi\) 側には入りません（ここは“数学的に決まる”）。

---

## 3. 実装（`improved1207_paper_jit.py`）との 1対1 対応

`improved1207_paper_jit.py` は、論文の強形式を **材料点 + implicit（\(\dot x\approx(x^{n+1}-x^n)/\Delta t\)）** にして、各ステップで Newton 法で解く構造です。

さらに数値的に \(\phi,\phi_0,\psi\in(0,1)\) を保つため **barrier法**（論文にも言及）を加えています。

### 3.1 主要対応（記号→コード）

- \(\phi\) / \(\phi_0\) / \(\psi\) / \(\gamma\) の状態ベクトル順序：コード冒頭の `State g (10,)` と一致
- \(\bar\phi=\phi\odot\psi\)：コードでは `phi_new * psi_new`
- \(A\bar\phi\)：コードでは `Interaction = A @ (phi_new * psi_new)`
- implicit Euler：コードでは `phidot=(phi_new-phi_old)/dt`, `psidot=(psi_new-psi_old)/dt`
- 体積制約：`Q[9] = np.sum(phi_new) + phi0_new - 1.0`
- バリア：`term1 = (Kp1*(2-4*v))/((v-1)^3*v^3)` 等（\(\phi,\phi_0,\psi\) に対して）

`Eta_vec[i]` は \(\eta_i\) に対応し、`Eta_phi_vec[i]` は論文の \(\frac12\dot\phi^\top\eta\dot\phi\) 由来の “\(\dot\phi\) 単独散逸” を **実装上は分離して扱う**ための係数です（論文は同じ \(\eta\) を使う書き方）。

---

## 4. ここまでで「数学的に言える結論」と限界

### 4.1 言えること（条件付きだが強い）

- `biofilm_simulation.pdf` の定義（\(\Psi, \Delta_s, f\)）を前提にすると、強形式(16)–(18)は変分計算から導ける（上のスケッチ）。
- `improved1207_paper_jit.py` の残差 `Q` は、(16)–(18) を **implicit Euler** で離散化し、さらに **barrier項**を加えたものに一致する。
- Newton法で各ステップ `Q=0` を解いているので、「離散化された方程式系の解を求めている」という意味で実装は妥当。

### 4.2 限界（ここは“証明”ではなく“前提・数値解析”の話）

- **barrier / clip** は解析モデル（連続方程式）そのものを修正するため、厳密には「論文の連続系をそのまま解いている」ではなく「論文の連続系を、数値的に可解・安定にするために修正した系」を解いている。
- 熱力学的一貫性（散逸不等式等）は、連続系では \(\eta_i>0\) 等の仮定で議論できるが、離散化やline-search等を含む数値法では **同値ではない**（ただし“破綻していないか”は数値検証できる）。

---

## 5. 実装妥当性を“自動チェック”する方針

最低限、以下はpytestで自動検査できます：

- **(i) 残差が論文式（離散化）と一致すること**
- **(ii) \(\psi\) 方程式が \(\gamma\) に依存しないこと**（制約が \(\phi\) のみなら数学的に必須）
- **(iii) 体積制約 `sum(phi)+phi0=1` が満たされること**（Newton解の精度）

この検査は `tmcmc/test_hamilton_model_consistency.py` に実装しています。

