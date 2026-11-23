# Dexbot-DynID

**Dexbot下肢闭链动力学参数辨识**

## 方法

傅里叶激励 · 闭链约束投影 · 惯性与摩擦回归 · QR基参数约简 · WLS · 可选LMI · 力矩验证。

## 安装

需要Python 3.8+、NumPy和SciPy；LMI优化可选装CVXPY。

```bash
python3 -m pip install -e .
python3 -m pytest -q
```

## 运行

```bash
python3 scripts/run_identification.py --csv trajectory.csv --inertia-npz regressor.npz --require-84 --output results
```

参数估计时添加 `--torque-csv torques.csv`。

**Drake：** 可选，用于从机器人模型构造闭链投影惯性回归矩阵；核心辨识程序不依赖Drake或机器人控制工程。

[方法说明](docs/METHOD_TO_CODE.md) · [接口说明](docs/REGRESSOR_INTERFACE.md)
