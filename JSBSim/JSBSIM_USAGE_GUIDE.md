# JSBSim 飞行数据获取指南

## 📖 简介

JSBSim 是一个开源的飞行动力学模型库，可以用来模拟各种飞机的飞行行为并获取详细的飞行数据。本指南将帮助您使用 Python 和 JSBSim 来获取飞机飞行过程中的各种数据。

## 🔧 安装 JSBSim

### 方法1：使用 pip 安装
```bash
pip install jsbsim
```

### 方法2：使用 conda 安装
```bash
conda install -c conda-forge jsbsim
```

### 方法3：从源码编译（高级用户）
```bash
git clone https://github.com/JSBSim-Team/jsbsim.git
cd jsbsim
python setup.py install
```

## 📦 依赖库安装

```bash
pip install pandas numpy
```

## 🚀 快速开始

### 1. 简单数据获取

使用 `simple_jsbsim_data.py` 进行基本的飞行数据获取：

```python
python simple_jsbsim_data.py
```

这个脚本提供了两种模式：
- **完整飞行数据获取**：收集一段时间内的所有飞行参数
- **实时数据监控**：实时显示当前飞行状态

### 2. 高级数据收集

使用 `jsbsim_data_collector.py` 进行更详细的数据收集和分析：

```python
python jsbsim_data_collector.py
```

## 📊 可获取的飞行数据

### 位置信息
- 纬度、经度
- 海拔高度
- 离地高度

### 速度信息
- 空速（节）
- 地速（节）
- 垂直速度（英尺/分钟）
- 机体坐标系速度分量

### 姿态信息
- 滚转角（度）
- 俯仰角（度）
- 航向角（度）
- 角速度

### 控制面位置
- 升降舵位置
- 副翼位置
- 方向舵位置
- 油门位置

### 发动机参数
- 发动机转速（RPM）
- 燃油流量

### 大气参数
- 气压高度
- 密度高度
- 温度
- 风速和风向

### 载荷因子
- G力
- 加速度分量

## 💻 代码示例

### 基本数据获取

```python
import jsbsim

# 创建JSBSim实例
fdm = jsbsim.FGFDMExec()

# 加载飞机模型
fdm.load_model('c172x')  # Cessna 172

# 设置初始条件
fdm.set_property_value('ic/h-sl-ft', 5000)  # 高度
fdm.set_property_value('ic/u-fps', 120)     # 速度

# 初始化
fdm.run_ic()

# 运行仿真并获取数据
for i in range(100):
    fdm.run()
    
    # 获取当前数据
    altitude = fdm.get_property_value('position/h-sl-ft')
    airspeed = fdm.get_property_value('velocities/vc-kts')
    
    print(f"高度: {altitude:.0f}ft, 空速: {airspeed:.1f}kt")
```

### 设置控制输入

```python
# 设置油门
fdm.set_property_value('fcs/throttle-cmd-norm', 0.8)  # 80%油门

# 设置升降舵
fdm.set_property_value('fcs/elevator-cmd-norm', 0.1)  # 轻微拉杆

# 设置副翼
fdm.set_property_value('fcs/aileron-cmd-norm', -0.2)  # 左滚

# 设置方向舵
fdm.set_property_value('fcs/rudder-cmd-norm', 0.1)    # 右舵
```

## 🛩️ 可用飞机模型

JSBSim 包含多种飞机模型：

- `c172x` - Cessna 172（小型单发飞机）
- `f16` - F-16 战斗机
- `737` - Boeing 737 客机
- `a320` - Airbus A320 客机
- `ball` - 简单的球体模型（用于测试）

## 📁 数据保存格式

### JSON 格式
```json
[
  {
    "时间(秒)": 0.0,
    "高度(英尺)": 5000.0,
    "空速(节)": 120.5,
    "纬度(度)": 37.0,
    "经度(度)": -122.0
  }
]
```

### CSV 格式
```csv
时间(秒),高度(英尺),空速(节),纬度(度),经度(度)
0.0,5000.0,120.5,37.0,-122.0
0.1,5001.2,120.8,37.0001,-122.0001
```

## 🔍 常见问题解决

### 1. JSBSim 安装失败

**问题**：`pip install jsbsim` 失败

**解决方案**：
- 确保 Python 版本 >= 3.6
- 尝试使用 conda 安装
- 检查网络连接
- 使用国内镜像：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple jsbsim`

### 2. 无法加载飞机模型

**问题**：`load_model()` 返回 False

**解决方案**：
- 检查模型名称是否正确
- 确保 JSBSim 数据文件已正确安装
- 尝试使用绝对路径设置数据目录

### 3. 数据获取异常

**问题**：获取的数据值异常或为 NaN

**解决方案**：
- 确保已调用 `run_ic()` 初始化
- 检查属性名称是否正确
- 确保仿真已开始运行

## 📚 进阶使用

### 自定义初始条件

```python
# 设置详细的初始条件
initial_conditions = {
    'ic/h-sl-ft': 10000,        # 高度
    'ic/long-gc-deg': -122.0,   # 经度
    'ic/lat-gc-deg': 37.0,      # 纬度
    'ic/u-fps': 200,            # 前向速度
    'ic/v-fps': 0,              # 侧向速度
    'ic/w-fps': 0,              # 垂直速度
    'ic/phi-deg': 0,            # 滚转角
    'ic/theta-deg': 5,          # 俯仰角
    'ic/psi-deg': 90,           # 航向角
}

for prop, value in initial_conditions.items():
    fdm.set_property_value(prop, value)
```

### 批量数据处理

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取保存的数据
df = pd.read_csv('jsbsim_flight_data_20231201_120000.csv')

# 绘制高度变化图
plt.figure(figsize=(10, 6))
plt.plot(df['时间(秒)'], df['高度(英尺)'])
plt.xlabel('时间 (秒)')
plt.ylabel('高度 (英尺)')
plt.title('飞行高度变化')
plt.grid(True)
plt.show()
```

## 🔗 相关资源

- [JSBSim 官方网站](https://jsbsim.sourceforge.net/)
- [JSBSim GitHub 仓库](https://github.com/JSBSim-Team/jsbsim)
- [JSBSim Python API 文档](https://jsbsim-team.github.io/jsbsim/)
- [飞行动力学基础知识](https://en.wikipedia.org/wiki/Flight_dynamics)

## 💡 提示

1. **性能优化**：对于长时间仿真，考虑调整时间步长和数据记录频率
2. **内存管理**：大量数据收集时注意内存使用，及时保存和清理数据
3. **单位转换**：注意 JSBSim 使用英制单位，需要时进行单位转换
4. **模型选择**：根据研究需求选择合适的飞机模型
5. **数据验证**：定期检查获取的数据是否合理，避免仿真发散

---

**祝您使用愉快！如有问题，请参考官方文档或社区支持。**