""" 
FlightGear机器学习自动驾驶系统
使用强化学习和神经网络来智能控制飞机起飞、飞行和着陆
"""
import time
import math
import numpy as np
import json
import os
from collections import deque
from flightgear_python.fg_if import TelnetConnection
from tqdm import tqdm

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TENSORFLOW = True
except ImportError:
    print("警告：未安装TensorFlow，将使用简化的机器学习模型")
    HAS_TENSORFLOW = False

class SimpleNeuralNetwork:
    """简化的神经网络实现（当TensorFlow不可用时）"""
    
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 初始化权重
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def predict(self, X):
        return self.forward(X)

class FlightState:
    """飞行状态数据结构"""
    
    def __init__(self):
        self.altitude = 0.0
        self.airspeed = 0.0
        self.heading = 0.0
        self.throttle = 0.0
        self.gear_down = True
        self.flaps = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.vertical_speed = 0.0
        self.ground_speed = 0.0
        
    def to_array(self):
        """转换为numpy数组用于机器学习"""
        return np.array([
            self.altitude / 10000.0,  # 归一化到0-1
            self.airspeed / 300.0,
            self.heading / 360.0,
            self.throttle,
            1.0 if self.gear_down else 0.0,
            self.flaps,
            (self.pitch + 90) / 180.0,  # 归一化到0-1
            (self.roll + 180) / 360.0,
            (self.vertical_speed + 5000) / 10000.0,
            self.ground_speed / 300.0
        ])
    
    def from_fg(self, fg_connection):
        """从FlightGear获取状态"""
        try:
            self.altitude = fg_connection.get_prop('/position/altitude-ft')
            self.airspeed = fg_connection.get_prop('/velocities/airspeed-kt')
            self.heading = fg_connection.get_prop('/orientation/heading-deg')
            self.throttle = fg_connection.get_prop('/controls/engines/engine[0]/throttle')
            self.gear_down = fg_connection.get_prop('/controls/gear/gear-down')
            self.flaps = fg_connection.get_prop('/controls/flight/flaps')
            self.pitch = fg_connection.get_prop('/orientation/pitch-deg')
            self.roll = fg_connection.get_prop('/orientation/roll-deg')
            self.vertical_speed = fg_connection.get_prop('/velocities/vertical-speed-fps')
            self.ground_speed = fg_connection.get_prop('/velocities/groundspeed-kt')
            return True
        except Exception as e:
            print(f"获取飞行状态失败: {e}")
            return False

class FlightAction:
    """飞行动作数据结构"""
    
    def __init__(self):
        self.throttle_delta = 0.0  # 油门变化量
        self.gear_action = 0  # 0=不变, 1=放下, -1=收起
        self.flaps_delta = 0.0  # 襟翼变化量
        
    def from_array(self, action_array):
        """从数组设置动作"""
        self.throttle_delta = (action_array[0] - 0.5) * 0.2  # -0.1到0.1
        self.gear_action = int(action_array[1] * 3) - 1  # -1, 0, 1
        self.flaps_delta = (action_array[2] - 0.5) * 0.2  # -0.1到0.1
        
    def apply_to_fg(self, fg_connection, current_state):
        """应用动作到FlightGear"""
        try:
            # 应用油门变化
            new_throttle = np.clip(current_state.throttle + self.throttle_delta, 0.0, 1.0)
            fg_connection.set_prop('/controls/engines/engine[0]/throttle', new_throttle)
            
            # 应用起落架动作
            if self.gear_action == 1:
                fg_connection.set_prop('/controls/gear/gear-down', True)
            elif self.gear_action == -1:
                fg_connection.set_prop('/controls/gear/gear-down', False)
            
            # 应用襟翼变化
            new_flaps = np.clip(current_state.flaps + self.flaps_delta, 0.0, 1.0)
            fg_connection.set_prop('/controls/flight/flaps', new_flaps)
            
            return True
        except Exception as e:
            print(f"应用动作失败: {e}")
            return False

class TakeoffAgent:
    """起飞专用智能代理"""
    
    def __init__(self):
        self.state_size = 10  # FlightState的特征数量
        self.action_size = 3  # FlightAction的动作数量
        self.memory = deque(maxlen=2000)
        self.learning_rate = 0.001
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        if HAS_TENSORFLOW:
            self.model = self._build_tf_model()
        else:
            self.model = SimpleNeuralNetwork(self.state_size, 64, self.action_size)
        
        # 起飞阶段定义
        self.takeoff_phases = {
            'ground': {'min_speed': 0, 'max_speed': 30, 'target_throttle': 0.3},
            'acceleration': {'min_speed': 30, 'max_speed': 60, 'target_throttle': 0.8},
            'rotation': {'min_speed': 60, 'max_speed': 80, 'target_throttle': 0.9},
            'climb': {'min_speed': 80, 'max_speed': 150, 'target_throttle': 0.8}
        }
        
    def _build_tf_model(self):
        """构建TensorFlow模型"""
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_size, activation='sigmoid')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                     loss='mse')
        return model
    
    def get_takeoff_phase(self, state):
        """根据当前状态判断起飞阶段"""
        speed = state.airspeed
        altitude = state.altitude
        
        if altitude > 500:  # 已经起飞完成
            return 'completed'
        elif speed >= 80:
            return 'climb'
        elif speed >= 60:
            return 'rotation'
        elif speed >= 30:
            return 'acceleration'
        else:
            return 'ground'
    
    def get_expert_action(self, state):
        """基于专家知识的动作（用于训练）"""
        phase = self.get_takeoff_phase(state)
        action = FlightAction()
        
        if phase == 'ground':
            # 地面阶段：逐渐增加油门，确保起落架放下
            if state.throttle < 0.3:
                action.throttle_delta = 0.1
            action.gear_action = 1  # 确保起落架放下
            if state.flaps < 0.3:
                action.flaps_delta = 0.1
                
        elif phase == 'acceleration':
            # 加速阶段：增加油门到0.8
            if state.throttle < 0.8:
                action.throttle_delta = 0.05
            action.gear_action = 1  # 保持起落架放下
            
        elif phase == 'rotation':
            # 拉升阶段：保持高油门，准备收起落架
            if state.throttle < 0.9:
                action.throttle_delta = 0.02
            if state.airspeed > 70:  # 速度足够时收起落架
                action.gear_action = -1
                
        elif phase == 'climb':
            # 爬升阶段：调整油门，收起襟翼
            if state.throttle > 0.8:
                action.throttle_delta = -0.02
            action.gear_action = -1  # 确保起落架收起
            if state.flaps > 0:
                action.flaps_delta = -0.05
        
        return action
    
    def act(self, state):
        """选择动作"""
        if np.random.rand() <= self.epsilon:
            # 探索：使用专家知识
            expert_action = self.get_expert_action(state)
            action_array = np.array([
                (expert_action.throttle_delta + 0.1) / 0.2 + 0.5,
                (expert_action.gear_action + 1) / 2,
                (expert_action.flaps_delta + 0.1) / 0.2 + 0.5
            ])
        else:
            # 利用：使用神经网络
            state_array = state.to_array().reshape(1, -1)
            if HAS_TENSORFLOW:
                action_array = self.model.predict(state_array, verbose=0)[0]
            else:
                action_array = self.model.predict(state_array)[0]
        
        action = FlightAction()
        action.from_array(action_array)
        return action
    
    def calculate_reward(self, prev_state, action, new_state):
        """计算奖励函数"""
        reward = 0.0
        
        # 基础奖励：保持飞行
        reward += 1.0
        
        # 速度奖励
        phase = self.get_takeoff_phase(new_state)
        if phase in self.takeoff_phases:
            target_range = self.takeoff_phases[phase]
            if target_range['min_speed'] <= new_state.airspeed <= target_range['max_speed']:
                reward += 5.0
            else:
                reward -= 2.0
        
        # 高度奖励
        if new_state.altitude > prev_state.altitude and new_state.airspeed > 60:
            reward += 10.0  # 成功爬升
        
        # 油门控制奖励
        if phase in self.takeoff_phases:
            target_throttle = self.takeoff_phases[phase]['target_throttle']
            throttle_diff = abs(new_state.throttle - target_throttle)
            reward += max(0, 2.0 - throttle_diff * 5)
        
        # 起落架控制奖励
        if new_state.airspeed < 70 and new_state.gear_down:
            reward += 2.0  # 低速时起落架应该放下
        elif new_state.airspeed > 80 and not new_state.gear_down:
            reward += 2.0  # 高速时起落架应该收起
        
        # 惩罚项
        if new_state.airspeed > 200:  # 速度过快
            reward -= 10.0
        if new_state.altitude < 0:  # 坠毁
            reward -= 100.0
        
        return reward
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=32):
        """经验回放训练"""
        if len(self.memory) < batch_size:
            return
        
        if not HAS_TENSORFLOW:
            # 简化版本的学习（不实现完整的经验回放）
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            return
        
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        states = np.array([self.memory[i][0].to_array() for i in batch])
        actions = np.array([self.memory[i][1] for i in batch])
        rewards = np.array([self.memory[i][2] for i in batch])
        next_states = np.array([self.memory[i][3].to_array() for i in batch])
        dones = np.array([self.memory[i][4] for i in batch])
        
        target = rewards + 0.95 * np.amax(self.model.predict(next_states, verbose=0), axis=1) * (1 - dones)
        target_full = self.model.predict(states, verbose=0)
        
        for i in range(batch_size):
            target_full[i] = actions[i]
        
        self.model.fit(states, target_full, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath):
        """保存模型"""
        if HAS_TENSORFLOW:
            self.model.save(filepath)
        else:
            # 保存简化模型的权重
            model_data = {
                'W1': self.model.W1.tolist(),
                'b1': self.model.b1.tolist(),
                'W2': self.model.W2.tolist(),
                'b2': self.model.b2.tolist(),
                'epsilon': self.epsilon
            }
            with open(filepath + '.json', 'w') as f:
                json.dump(model_data, f)
    
    def load_model(self, filepath):
        """加载模型"""
        if HAS_TENSORFLOW and os.path.exists(filepath):
            self.model = keras.models.load_model(filepath)
        elif os.path.exists(filepath + '.json'):
            with open(filepath + '.json', 'r') as f:
                model_data = json.load(f)
            self.model.W1 = np.array(model_data['W1'])
            self.model.b1 = np.array(model_data['b1'])
            self.model.W2 = np.array(model_data['W2'])
            self.model.b2 = np.array(model_data['b2'])
            self.epsilon = model_data.get('epsilon', 0.1)

class MLFlightGearAutopilot:
    """机器学习FlightGear自动驾驶系统"""
    
    def __init__(self, host='localhost', port=5500):
        self.fg = TelnetConnection(host, port)
        self.connected = False
        self.takeoff_agent = TakeoffAgent()
        
        # 飞行计划
        self.departure_point = None
        self.destination_point = None
        self.cruise_altitude = 10000
        
        # 训练参数
        self.training_mode = True
        self.episode_count = 0
        self.max_episodes = 100
        
    def connect(self):
        """连接到FlightGear"""
        try:
            print(f"正在连接到FlightGear Telnet服务器: {self.fg.host}:{self.fg.port}")
            self.fg.connect()
            test_prop = self.fg.get_prop('/sim/aircraft')
            self.connected = True
            print(f"成功连接到FlightGear！当前飞机: {test_prop}")
            return True
        except Exception as e:
            print(f"连接FlightGear失败: {e}")
            print("请确保FlightGear正在运行并启用了Telnet服务器")
            self.connected = False
            return False
    
    def set_flight_plan(self, departure_lat, departure_lon, departure_alt,
                       destination_lat, destination_lon, destination_alt, cruise_alt=10000):
        """设置飞行计划"""
        self.departure_point = (departure_lat, departure_lon, departure_alt)
        self.destination_point = (destination_lat, destination_lon, destination_alt)
        self.cruise_altitude = cruise_alt
        
        distance = self.calculate_distance(
            departure_lat, departure_lon, destination_lat, destination_lon
        )
        
        print(f"飞行计划已设置:")
        print(f"起点: {departure_lat:.6f}, {departure_lon:.6f}, {departure_alt} 英尺")
        print(f"终点: {destination_lat:.6f}, {destination_lon:.6f}, {destination_alt} 英尺")
        print(f"距离: {distance:.2f}km")
        print(f"巡航高度: {cruise_alt} 英尺")
    
    def initialize_aircraft(self):
        """初始化飞机状态"""
        if not self.connected or not self.departure_point:
            return False
        
        try:
            departure = self.departure_point
            
            # 设置位置
            self.fg.set_prop('/position/latitude-deg', departure[0])
            self.fg.set_prop('/position/longitude-deg', departure[1])
            self.fg.set_prop('/position/altitude-ft', departure[2])
            
            # 设置初始状态
            self.fg.set_prop('/controls/engines/engine[0]/throttle', 0.0)
            self.fg.set_prop('/controls/gear/gear-down', True)
            self.fg.set_prop('/velocities/airspeed-kt', 0)
            self.fg.set_prop('/controls/flight/flaps', 0.3)
            
            # 禁用自动驾驶
            self.fg.set_prop('/autopilot/locks/autopilot', '')
            
            time.sleep(2)  # 等待设置生效
            print("飞机初始化完成")
            return True
            
        except Exception as e:
            print(f"初始化飞机失败: {e}")
            return False
    
    def execute_ml_takeoff(self, max_steps=1000):
        """执行机器学习控制的起飞"""
        if not self.connected:
            print("未连接到FlightGear")
            return False
        
        print("\n开始机器学习控制的起飞过程...")
        print("="*50)
        
        # 初始化
        if not self.initialize_aircraft():
            return False
        
        current_state = FlightState()
        if not current_state.from_fg(self.fg):
            return False
        
        start_altitude = current_state.altitude
        target_altitude = start_altitude + 500  # 起飞目标高度
        
        print(f"起始高度: {start_altitude:.1f} 英尺")
        print(f"目标高度: {target_altitude:.1f} 英尺")
        
        step = 0
        takeoff_completed = False
        
        # 使用tqdm显示进度
        with tqdm(total=max_steps, desc="ML起飞进度", unit="步") as pbar:
            
            while step < max_steps and not takeoff_completed:
                # 获取当前状态
                prev_state = current_state
                current_state = FlightState()
                if not current_state.from_fg(self.fg):
                    break
                
                # 智能代理选择动作
                action = self.takeoff_agent.act(current_state)
                
                # 应用动作
                if not action.apply_to_fg(self.fg, current_state):
                    break
                
                # 等待动作生效
                time.sleep(0.5)
                
                # 获取新状态
                new_state = FlightState()
                if not new_state.from_fg(self.fg):
                    break
                
                # 计算奖励
                reward = self.takeoff_agent.calculate_reward(prev_state, action, new_state)
                
                # 检查是否完成起飞
                if new_state.altitude >= target_altitude and new_state.airspeed > 80:
                    takeoff_completed = True
                    reward += 50  # 完成奖励
                
                # 存储经验（如果在训练模式）
                if self.training_mode:
                    self.takeoff_agent.remember(current_state, action, reward, new_state, takeoff_completed)
                
                # 更新进度条
                phase = self.takeoff_agent.get_takeoff_phase(new_state)
                altitude_progress = min(100, (new_state.altitude - start_altitude) / (target_altitude - start_altitude) * 100)
                
                pbar.set_description(f"ML起飞 [{phase}] 高度:{new_state.altitude:.0f}ft 速度:{new_state.airspeed:.0f}kt 进度:{altitude_progress:.1f}%")
                pbar.update(1)
                
                current_state = new_state
                step += 1
                
                # 安全检查
                if new_state.altitude < start_altitude - 100:  # 高度损失过多
                    print("\n警告：高度损失过多，中止起飞")
                    break
        
        # 训练神经网络
        if self.training_mode and len(self.takeoff_agent.memory) > 32:
            print("\n训练神经网络...")
            self.takeoff_agent.replay()
        
        if takeoff_completed:
            print(f"\n起飞成功完成！")
            print(f"最终高度: {current_state.altitude:.1f} 英尺")
            print(f"最终速度: {current_state.airspeed:.1f} 节")
            print(f"总步数: {step}")
            return True
        else:
            print(f"\n起飞未完成，已执行 {step} 步")
            return False
    
    def train_takeoff_agent(self, episodes=50):
        """训练起飞代理"""
        print(f"\n开始训练起飞代理，共 {episodes} 轮...")
        
        success_count = 0
        
        for episode in range(episodes):
            print(f"\n第 {episode + 1}/{episodes} 轮训练")
            
            # 重置环境
            if not self.initialize_aircraft():
                continue
            
            # 执行起飞
            success = self.execute_ml_takeoff(max_steps=500)
            if success:
                success_count += 1
            
            # 保存模型
            if (episode + 1) % 10 == 0:
                model_path = f"takeoff_model_episode_{episode + 1}"
                self.takeoff_agent.save_model(model_path)
                print(f"模型已保存: {model_path}")
        
        success_rate = success_count / episodes * 100
        print(f"\n训练完成！成功率: {success_rate:.1f}% ({success_count}/{episodes})")
        
        # 保存最终模型
        self.takeoff_agent.save_model("takeoff_model_final")
        print("最终模型已保存: takeoff_model_final")
    
    def load_trained_model(self, model_path="takeoff_model_final"):
        """加载训练好的模型"""
        try:
            self.takeoff_agent.load_model(model_path)
            self.training_mode = False  # 切换到推理模式
            print(f"已加载训练模型: {model_path}")
            return True
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False
    
    @staticmethod
    def calculate_distance(lat1, lon1, lat2, lon2):
        """计算两点间的距离（km）"""
        R = 6371  # 地球半径
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat_rad = math.radians(lat2 - lat1)
        dlon_rad = math.radians(lon2 - lon1)
        
        a = math.sin(dlat_rad/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon_rad/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def stop(self):
        """断开连接"""
        if self.connected:
            try:
                self.fg.close()
                self.connected = False
                print("FlightGear连接已断开")
            except Exception as e:
                print(f"断开连接时出错: {e}")

# 预设飞行路线
FLIGHT_ROUTES = {
    'BIKF到BGBW': {
        'departure': (64.1300, -21.9406, 171),
        'destination': (61.1572, -45.4258, 283),
        'cruise_alt': 25000
    },
    'PANC到PAFA': {
        'departure': (61.1744, -149.9961, 152),
        'destination': (64.8378, -147.8564, 434),
        'cruise_alt': 20000
    },
    'PHNL到PHOG': {
        'departure': (21.3187, -157.9224, 13),
        'destination': (20.8987, -156.4306, 80),
        'cruise_alt': 15000
    }
}

def main():
    print("FlightGear机器学习自动驾驶系统")
    print("="*60)
    print("本系统使用机器学习技术智能控制飞机起飞")
    print("支持训练模式和推理模式")
    print("="*60)
    
    # 创建自动驾驶系统
    autopilot = MLFlightGearAutopilot()
    
    try:
        # 连接FlightGear
        if not autopilot.connect():
            print("\n无法连接到FlightGear，请检查：")
            print("1. FlightGear是否正在运行")
            print("2. 是否启用了Telnet服务器 (--telnet=socket,bi,60,localhost,5500,tcp)")
            return
        
        # 选择模式
        print("\n请选择模式：")
        print("1. 训练模式 - 训练起飞代理")
        print("2. 推理模式 - 使用训练好的模型执行起飞")
        print("3. 测试模式 - 使用专家知识执行起飞")
        
        mode = input("请选择模式 (1/2/3): ").strip()
        
        # 设置飞行计划
        print("\n可用飞行路线：")
        for i, route_name in enumerate(FLIGHT_ROUTES.keys(), 1):
            print(f"{i}. {route_name}")
        
        route_choice = input("请选择路线 (1-3): ").strip()
        route_names = list(FLIGHT_ROUTES.keys())
        
        if route_choice.isdigit() and 1 <= int(route_choice) <= len(route_names):
            route_name = route_names[int(route_choice) - 1]
            route_info = FLIGHT_ROUTES[route_name]
            
            departure = route_info['departure']
            destination = route_info['destination']
            cruise_alt = route_info['cruise_alt']
            
            autopilot.set_flight_plan(
                departure[0], departure[1], departure[2],
                destination[0], destination[1], destination[2],
                cruise_alt
            )
            
            print(f"\n已选择路线: {route_name}")
        else:
            print("无效选择，使用默认路线")
            route_info = FLIGHT_ROUTES['PHNL到PHOG']
            departure = route_info['departure']
            destination = route_info['destination']
            autopilot.set_flight_plan(
                departure[0], departure[1], departure[2],
                destination[0], destination[1], destination[2],
                route_info['cruise_alt']
            )
        
        # 执行相应模式
        if mode == '1':
            # 训练模式
            episodes = input("请输入训练轮数 (默认50): ").strip()
            episodes = int(episodes) if episodes.isdigit() else 50
            autopilot.train_takeoff_agent(episodes)
            
        elif mode == '2':
            # 推理模式
            model_path = input("请输入模型路径 (默认: takeoff_model_final): ").strip()
            model_path = model_path if model_path else "takeoff_model_final"
            
            if autopilot.load_trained_model(model_path):
                print("\n使用训练好的模型执行起飞...")
                input("按回车键开始...")
                autopilot.execute_ml_takeoff()
            else:
                print("模型加载失败，切换到测试模式")
                autopilot.training_mode = False
                autopilot.takeoff_agent.epsilon = 0.0  # 完全使用专家知识
                input("按回车键开始...")
                autopilot.execute_ml_takeoff()
                
        else:
            # 测试模式
            autopilot.training_mode = False
            autopilot.takeoff_agent.epsilon = 1.0  # 完全使用专家知识
            print("\n使用专家知识执行起飞...")
            input("按回车键开始...")
            autopilot.execute_ml_takeoff()
        
        print("\n程序执行完毕！")
        
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        autopilot.stop()
        print("\n程序结束，再见！")

if __name__ == '__main__':
    main()