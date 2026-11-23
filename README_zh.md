# Dexbot-DynID

**Dexbot下肢动力学参数辨识**

对应论文 *Identification of lower-limb dynamic parameters in a humanoid robot based on closed-loop ankle constraint projections* 的Python辨识方法实现。

## 方法

傅里叶激励 · 闭链约束投影 · 惯性与驱动摩擦回归 · QR基参数提取 · 加权最小二乘（WLS） · 可选LMI物理一致性约束 · 力矩预测与验证。

## 依赖

- **Python 3.8+**、NumPy、SciPy
- **CVXPY及SDP求解器**：可选，仅用于LMI
- **Drake**：可选，用于根据机器人模型生成动力学回归矩阵；**核心辨识代码不依赖Drake**

Drake可用于读取URDF、计算机器人运动学和惯性动力学。需要由模型相关的外部程序处理踝部闭链约束并导出六驱动坐标下的惯性回归矩阵。取得矩阵后，QR、WLS、LMI无需Drake或原机器人控制工程。

## 所需数据

1. **关节运动轨迹：** 六驱动的位置、速度、加速度和时间，可采用一个15秒周期。
2. **惯性回归矩阵：** 与轨迹逐点对应的 `[N,6,60]` 数组，程序加入24列驱动附加项，构成84列扩展回归矩阵。
3. **驱动力矩：** WLS/LMI拟合时提供；仅做QR无需力矩。独立验证需另一组轨迹、回归矩阵和力矩数据。

驱动顺序为髋Pitch、髋Roll、髋Yaw、膝、左踝LONG摇臂、左踝SHORT摇臂。详见[数据格式](data/INPUT_FORMAT.md)和[模型接口](docs/REGRESSOR_INTERFACE.md)。

## 安装与运行

```bash
python3 -m pip install -e .
python3 scripts/check_data.py --csv trajectory.csv
python3 scripts/run_identification.py --csv trajectory.csv --inertia-npz inertia_regressor.npz --require-84 --output results
```

WLS可添加 `--torque-csv torques.csv`；LMI需先安装 `pip install -e '.[sdp]'`，再添加 `--lmi --n-links 6`。测试使用 `pip install -e '.[test]'` 和 `python3 -m pytest -q`。

基参数数量由输入回归矩阵的数值秩确定，不在代码中固定为论文报告的59。

**相关说明：** [论文方法对应](docs/METHOD_TO_CODE.md) · [Drake接口](docs/DRAKE_AND_DATA.md) · [GitHub发布](docs/GITHUB_PUBLISH.md)
