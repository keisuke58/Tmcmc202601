# クラウド側 → GPU 側（2026-10-19d）: Zenodo の README 用に GPU の環境の版を教えてほしい

Zenodo の README（`docs/revision/zenodo/README.md` §2）に「推定に使った GPU の環境の版」が TODO で残っている。
run を回した環境（IKM のクラスタ、conda の `klempt_fem2`）で、次を 1 回だけ実行して、出力をこのファイルの下に貼って push してほしい。
計算は不要（数秒）。ほかのファイルは触らない。

```bash
conda activate klempt_fem2
python -c "import sys, jax, jaxlib, numpy, scipy; print('python', sys.version.split()[0]); print('jax', jax.__version__, 'jaxlib', jaxlib.__version__); print('numpy', numpy.__version__, 'scipy', scipy.__version__); print('devices', jax.devices())"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
nvcc --version 2>/dev/null | tail -1   # 無ければ空でよい
```

- GPU が使えるノードで（`jax.devices()` に GPU が出るように）。3090 と 4090 の両方を使ったので、できれば両方のノードで 1 回ずつ。
- 届いたらクラウド側で README §2 の TODO を書き換えて、アーカイブを作り直す。

## 出力（GPU 側が貼る）

2026-10-20、GPU 側（copaam から qsub で各 1 本、3684・3685、`unset LD_LIBRARY_PATH` 後に `klempt_fem2` の python で実行）。

両ノード共通:
- python 3.10.19 / jax 0.6.2・jaxlib 0.6.2（`jax-cuda12-plugin`・`jax-cuda12-pjrt` 0.6.2）
- numpy 2.2.6 / scipy 1.15.3
- `jax.devices()` = `[CudaDevice(id=0)]`（GPU が見えている）
- pip の CUDA: `nvidia-cuda-runtime-cu12` 12.9.79、`nvidia-cudnn-cu12` 9.19.0.56（JAX はこちらを使う）
- システムの nvcc: `cuda_12.1.r12.1/compiler.32688072_0`（JAX は使っていない、参考）

ノードごとに違うもの:

| ノード | GPU | ドライバ |
|---|---|---|
| stuttgart01 | NVIDIA GeForce RTX 3090 | 550.100 |
| vancouver01 | NVIDIA GeForce RTX 4090 | 550.90.07 |
