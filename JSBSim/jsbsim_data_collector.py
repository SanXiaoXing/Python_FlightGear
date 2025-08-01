#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSBSim 飞行数据采集器
用于从JSBSim飞行动力学模型中获取飞机飞行过程中的各种数据
"""

import jsbsim
import time
import json
import csv
import os
from datetime import datetime
import numpy as np

class JSBSimDataCollector:
    def __init__(self, aircraft_model="f16", dt=0.01):
        """
        初始化JSBSim数据采集器
        
        Args:
            aircraft_model: 飞机模型名称
            dt: 仿真时间步长（秒）
        """
        self.fdm = jsbsim.FGFDMExec(None)
        self.aircraft_model = aircraft_model
        self.dt = dt
        self.data_log = []
        self.is_running = False
        
        # 设置JSBSim路径（如果需要）
        # self.fdm.set_aircraft_path('path/to/aircraft')
        # self.fdm.set_engine_path('path/to/engines')
        # self.fdm.set_systems_path('path/to/systems')
        
    def initialize_aircraft(self, initial_conditions=None):
        """
        初始化飞机和初始条件
        
        Args:
            initial_conditions: 初始条件字典
        """
        try:
            # 加载飞机模型
            if not self.fdm.load_model(self.aircraft_model):
                raise Exception(f"无法加载飞机模型: {self.aircraft_model}")
            
            # 设置时间步长
            self.fdm.set_dt(self.dt)
            
            # 设置初始条件
            if initial_conditions is None:
                initial_conditions = {
                    'ic/h-sl-ft': 5000,        # 海拔高度（英尺）
                    'ic/long-gc-deg': -122.0,  # 经度
                    'ic/lat-gc-deg': 37.0,     # 纬度
                    'ic/u-fps': 100,           # 前向速度（英尺/秒）
                    'ic/v-fps': 0,             # 侧向速度
                    'ic/w-fps': 0,             # 垂直速度
                    'ic/phi-deg': 0,           # 滚转角
                    'ic/theta-deg': 0,         # 俯仰角
                    'ic/psi-deg': 0,           # 偏航角
                }
            
            # 应用初始条件
            for prop, value in initial_conditions.items():
                self.fdm.set_property_value(prop, value)
            
            # 初始化模型
            if not self.fdm.run_ic():
                raise Exception("初始化失败")
            
            print(f"✅ 成功初始化飞机模型: {self.aircraft_model}")
            return True
            
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            return False
    
    def get_flight_data(self):
        """
        获取当前飞行数据
        
        Returns:
            dict: 包含各种飞行参数的字典
        """
        data = {
            # 时间
            'time': self.fdm.get_sim_time(),
            
            # 位置信息
            'latitude': self.fdm.get_property_value('position/lat-gc-deg'),
            'longitude': self.fdm.get_property_value('position/long-gc-deg'),
            'altitude_ft': self.fdm.get_property_value('position/h-sl-ft'),
            'altitude_agl_ft': self.fdm.get_property_value('position/h-agl-ft'),
            
            # 速度信息
            'airspeed_kt': self.fdm.get_property_value('velocities/vc-kts'),
            'groundspeed_kt': self.fdm.get_property_value('velocities/vg-fps') * 0.592484,  # 转换为节
            'vertical_speed_fpm': self.fdm.get_property_value('velocities/h-dot-fps') * 60,  # 英尺/分钟
            'u_fps': self.fdm.get_property_value('velocities/u-fps'),  # 机体坐标系速度
            'v_fps': self.fdm.get_property_value('velocities/v-fps'),
            'w_fps': self.fdm.get_property_value('velocities/w-fps'),
            
            # 姿态信息
            'roll_deg': self.fdm.get_property_value('attitude/phi-deg'),
            'pitch_deg': self.fdm.get_property_value('attitude/theta-deg'),
            'heading_deg': self.fdm.get_property_value('attitude/psi-deg'),
            
            # 角速度
            'roll_rate_dps': self.fdm.get_property_value('velocities/p-rad_sec') * 57.2958,  # 度/秒
            'pitch_rate_dps': self.fdm.get_property_value('velocities/q-rad_sec') * 57.2958,
            'yaw_rate_dps': self.fdm.get_property_value('velocities/r-rad_sec') * 57.2958,
            
            # 控制面位置
            'elevator_pos': self.fdm.get_property_value('fcs/elevator-pos-norm'),
            'aileron_pos': self.fdm.get_property_value('fcs/aileron-pos-norm'),
            'rudder_pos': self.fdm.get_property_value('fcs/rudder-pos-norm'),
            'throttle_pos': self.fdm.get_property_value('fcs/throttle-pos-norm'),
            
            # 发动机参数
            'engine_rpm': self.fdm.get_property_value('propulsion/engine/engine-rpm'),
            'fuel_flow_pph': self.fdm.get_property_value('propulsion/engine/fuel-flow-rate-pps') * 3600,  # 磅/小时
            
            # 大气参数
            'pressure_alt_ft': self.fdm.get_property_value('atmosphere/pressure-altitude'),
            'density_alt_ft': self.fdm.get_property_value('atmosphere/density-altitude'),
            'temperature_R': self.fdm.get_property_value('atmosphere/T-R'),  # 兰金度
            'wind_speed_fps': self.fdm.get_property_value('atmosphere/wind-mag-fps'),
            'wind_direction_deg': self.fdm.get_property_value('atmosphere/wind-dir-deg'),
            
            # 载荷因子
            'load_factor': self.fdm.get_property_value('accelerations/n-pilot-z-norm'),
            'acceleration_x': self.fdm.get_property_value('accelerations/udot-ft_sec2'),
            'acceleration_y': self.fdm.get_property_value('accelerations/vdot-ft_sec2'),
            'acceleration_z': self.fdm.get_property_value('accelerations/wdot-ft_sec2'),
        }
        
        return data
    
    def set_controls(self, elevator=None, aileron=None, rudder=None, throttle=None):
        """
        设置飞机控制输入
        
        Args:
            elevator: 升降舵位置 (-1 到 1)
            aileron: 副翼位置 (-1 到 1)
            rudder: 方向舵位置 (-1 到 1)
            throttle: 油门位置 (0 到 1)
        """
        if elevator is not None:
            self.fdm.set_property_value('fcs/elevator-cmd-norm', elevator)
        if aileron is not None:
            self.fdm.set_property_value('fcs/aileron-cmd-norm', aileron)
        if rudder is not None:
            self.fdm.set_property_value('fcs/rudder-cmd-norm', rudder)
        if throttle is not None:
            self.fdm.set_property_value('fcs/throttle-cmd-norm', throttle)
    
    def run_simulation(self, duration_seconds=60, log_interval=0.1, save_data=True):
        """
        运行仿真并收集数据
        
        Args:
            duration_seconds: 仿真持续时间（秒）
            log_interval: 数据记录间隔（秒）
            save_data: 是否保存数据到文件
        """
        print(f"🚁 开始仿真，持续时间: {duration_seconds}秒")
        
        self.data_log = []
        self.is_running = True
        start_time = time.time()
        last_log_time = 0
        
        try:
            while self.is_running and (time.time() - start_time) < duration_seconds:
                # 运行一个仿真步骤
                if not self.fdm.run():
                    print("❌ 仿真运行失败")
                    break
                
                # 记录数据
                current_sim_time = self.fdm.get_sim_time()
                if current_sim_time - last_log_time >= log_interval:
                    data = self.get_flight_data()
                    self.data_log.append(data)
                    last_log_time = current_sim_time
                    
                    # 显示进度
                    if len(self.data_log) % 50 == 0:  # 每50个数据点显示一次
                        print(f"📊 时间: {current_sim_time:.1f}s, 高度: {data['altitude_ft']:.0f}ft, 速度: {data['airspeed_kt']:.1f}kt")
                
                # 控制仿真速度（可选）
                time.sleep(0.001)
            
            print(f"✅ 仿真完成，共收集 {len(self.data_log)} 个数据点")
            
            if save_data and self.data_log:
                self.save_data_to_files()
            
        except KeyboardInterrupt:
            print("\n⏹️ 仿真被用户中断")
            if save_data and self.data_log:
                self.save_data_to_files()
        
        self.is_running = False
    
    def save_data_to_files(self):
        """
        保存数据到JSON和CSV文件
        """
        if not self.data_log:
            print("❌ 没有数据可保存")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存为JSON
        json_filename = f"flight_data_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(self.data_log, f, indent=2, ensure_ascii=False)
        print(f"💾 数据已保存到: {json_filename}")
        
        # 保存为CSV
        csv_filename = f"flight_data_{timestamp}.csv"
        if self.data_log:
            fieldnames = self.data_log[0].keys()
            with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.data_log)
            print(f"📊 数据已保存到: {csv_filename}")
    
    def analyze_data(self):
        """
        分析收集的飞行数据
        """
        if not self.data_log:
            print("❌ 没有数据可分析")
            return
        
        print("\n📈 飞行数据分析:")
        print("=" * 50)
        
        # 提取数据
        times = [d['time'] for d in self.data_log]
        altitudes = [d['altitude_ft'] for d in self.data_log]
        airspeeds = [d['airspeed_kt'] for d in self.data_log]
        
        # 基本统计
        print(f"仿真时长: {max(times):.1f} 秒")
        print(f"最大高度: {max(altitudes):.0f} 英尺")
        print(f"最小高度: {min(altitudes):.0f} 英尺")
        print(f"平均高度: {np.mean(altitudes):.0f} 英尺")
        print(f"最大空速: {max(airspeeds):.1f} 节")
        print(f"最小空速: {min(airspeeds):.1f} 节")
        print(f"平均空速: {np.mean(airspeeds):.1f} 节")
        
        # 高度变化率
        if len(altitudes) > 1:
            alt_changes = np.diff(altitudes)
            time_diffs = np.diff(times)
            climb_rates = alt_changes / time_diffs * 60  # 英尺/分钟
            print(f"最大爬升率: {max(climb_rates):.0f} 英尺/分钟")
            print(f"最大下降率: {min(climb_rates):.0f} 英尺/分钟")

def demo_basic_flight():
    """
    演示基本飞行数据收集
    """
    print("🛩️ JSBSim 飞行数据收集演示")
    print("=" * 40)
    
    # 创建数据收集器
    collector = JSBSimDataCollector(aircraft_model="f16")
    
    # 初始化飞机
    if not collector.initialize_aircraft():
        return
    
    # 设置基本控制（平飞）
    collector.set_controls(throttle=0.7, elevator=0.0, aileron=0.0, rudder=0.0)
    
    # 运行仿真
    collector.run_simulation(duration_seconds=30, log_interval=0.1)
    
    # 分析数据
    collector.analyze_data()

def demo_controlled_flight():
    """
    演示带控制输入的飞行
    """
    print("🎮 带控制输入的飞行演示")
    print("=" * 40)
    
    collector = JSBSimDataCollector(aircraft_model="f16")
    
    # 设置更合适的初始条件
    initial_conditions = {
        'ic/h-sl-ft': 0,        # 降低初始高度
        'ic/long-gc-deg': -122.0,
        'ic/lat-gc-deg': 37.0,
        'ic/u-fps': 120,           # 增加初始速度
        'ic/v-fps': 0,
        'ic/w-fps': 0,
        'ic/phi-deg': 0,
        'ic/theta-deg': 0,         # 水平姿态
        'ic/psi-deg': 0,
    }
    
    if not collector.initialize_aircraft(initial_conditions):
        return
    
    print("🚁 执行爬升机动...")
    
    # 手动控制仿真
    duration = 30  # 增加仿真时间
    dt = 0.1
    steps = int(duration / dt)
    
    for i in range(steps):
        sim_time = i * dt
        
        # 修正的动态控制输入
        if sim_time < 3:
            # 起始阶段：建立稳定飞行
            throttle = 0.85
            elevator = -0.05  # 先保持水平
        elif sim_time < 8:
            # 爬升准备阶段：增加油门
            throttle = 0.95
            elevator = -0.1  # 轻微下压建立速度
        elif sim_time < 18:
            # 爬升阶段：拉杆爬升
            throttle = 1
            elevator = -0.3   # 负值表示拉杆（向上）
        elif sim_time < 25:
            # 转平阶段：减小拉杆
            throttle = 0.85
            elevator = -0.1  # 轻微拉杆保持高度
        else:
            # 平飞阶段
            throttle = 0.75
            elevator = 0.0
        
        collector.set_controls(throttle=throttle, elevator=elevator)
        
        if not collector.fdm.run():
            break
        
        # 记录数据（更频繁的记录）
        if i % 5 == 0:  # 每0.5秒记录一次
            data = collector.get_flight_data()
            collector.data_log.append(data)
            
            # 显示详细信息
            climb_rate = data['vertical_speed_fpm']
            pitch = data['pitch_deg']
            altitude = data['altitude_ft']
            speed = data['airspeed_kt']
            print(f"时间: {sim_time:.1f}s, 高度: {altitude:.0f}ft, 速度: {speed:.1f}kt, 爬升率: {climb_rate:.0f}fpm, 俯仰: {pitch:.1f}°")
    collector.save_data_to_files()
    collector.analyze_data()
    
    # 额外分析爬升性能
    if collector.data_log:
        print("\n🔍 爬升性能分析:")
        print("=" * 30)
        
        initial_alt = collector.data_log[0]['altitude_ft']
        final_alt = collector.data_log[-1]['altitude_ft']
        alt_gain = final_alt - initial_alt
        
        max_climb_rate = max([d['vertical_speed_fpm'] for d in collector.data_log])
        min_climb_rate = min([d['vertical_speed_fpm'] for d in collector.data_log])
        
        print(f"初始高度: {initial_alt:.0f} 英尺")
        print(f"最终高度: {final_alt:.0f} 英尺")
        print(f"高度增益: {alt_gain:.0f} 英尺")
        print(f"最大爬升率: {max_climb_rate:.0f} 英尺/分钟")
        print(f"最大下降率: {min_climb_rate:.0f} 英尺/分钟")
        
        if alt_gain > 100:
            print("✅ 爬升成功！")
        elif alt_gain > 0:
            print("⚠️ 轻微爬升")
        else:
            print("❌ 爬升失败，飞机在下降")

def demo_complete_flight_cycle():
    """
    演示完整的飞行周期：起飞 -> 飞行 -> 降落
    """
    print("\n🛫 开始完整飞行周期演示")
    print("=" * 50)
    
    collector = JSBSimDataCollector()
    
    # 设置起飞初始条件（地面）
    initial_conditions = {
        'ic/h-sl-ft': 0,         # 起始高度100英尺（接近地面）
        'ic/long-gc-deg': -122.0,  # 经度
        'ic/lat-gc-deg': 37.0,     # 纬度
        'ic/u-fps': 30,            # 初始前向速度30英尺/秒（约20节）
        'ic/v-fps': 0,             # 侧向速度
        'ic/w-fps': 0,             # 垂直速度
        'ic/phi-deg': 0,           # 滚转角
        'ic/theta-deg': 0,         # 俯仰角
        'ic/psi-deg': 0,           # 偏航角
    }
    
    if not collector.initialize_aircraft(initial_conditions):
        return
    
    print("\n🚀 第一阶段：起飞过程")
    print("-" * 30)
    
    # 起飞阶段（0-60秒）
    takeoff_duration = 60
    for i in range(takeoff_duration * 100):  # 0.01秒步长
        sim_time = i * 0.01
        
        # 起飞控制逻辑
        if sim_time < 20:  # 前20秒：地面滑跑加速
            throttle = 1.0  # 全油门
            elevator = 0.0  # 保持水平
            print(f"地面滑跑阶段 - 时间: {sim_time:.1f}s")
        elif sim_time < 35:  # 20-35秒：抬轮起飞
            throttle = 1.0
            elevator = -0.1  # 轻微拉杆
            if sim_time == 20:
                print("开始抬轮起飞...")
        else:  # 35秒后：爬升
            throttle = 0.9
            elevator = -0.15  # 保持爬升姿态
            if sim_time == 35:
                print("进入爬升阶段...")
        
        collector.set_controls(elevator=elevator, throttle=throttle)
        
        if not collector.fdm.run():
            break
        
        # 记录数据
        if i % 10 == 0:  # 每0.1秒记录一次
            data = collector.get_flight_data()
            collector.data_log.append(data)
            
            if i % 100 == 0:  # 每秒显示一次状态
                print(f"高度: {data['altitude_ft']:.0f}ft, 速度: {data['airspeed_kt']:.1f}kt, 爬升率: {data['vertical_speed_fpm']:.0f}fpm")
    
    print("\n✈️ 第二阶段：巡航飞行")
    print("-" * 30)
    
    # 巡航阶段（60-180秒）
    cruise_duration = 120
    target_altitude = collector.get_flight_data()['altitude_ft']
    
    for i in range(cruise_duration * 100):
        sim_time = 60 + i * 0.01
        current_data = collector.get_flight_data()
        current_alt = current_data['altitude_ft']
        
        # 巡航控制逻辑
        throttle = 0.7  # 巡航油门
        
        # 高度保持控制
        alt_error = target_altitude - current_alt
        if alt_error > 50:
            elevator = -0.05  # 轻微爬升
        elif alt_error < -50:
            elevator = 0.05   # 轻微下降
        else:
            elevator = 0.0    # 保持水平
        
        collector.set_controls(elevator=elevator, throttle=throttle)
        
        if not collector.fdm.run():
            break
        
        # 记录数据
        if i % 10 == 0:
            data = collector.get_flight_data()
            collector.data_log.append(data)
            
            if i % 500 == 0:  # 每5秒显示一次状态
                print(f"巡航中 - 高度: {data['altitude_ft']:.0f}ft, 速度: {data['airspeed_kt']:.1f}kt")
    
    print("\n🛬 第三阶段：降落过程")
    print("-" * 30)
    
    # 降落阶段（180-300秒）
    landing_duration = 120
    
    for i in range(landing_duration * 100):
        sim_time = 180 + i * 0.01
        current_data = collector.get_flight_data()
        current_alt = current_data['altitude_ft']
        
        # 降落控制逻辑
        if sim_time < 220:  # 前40秒：开始下降
            throttle = 0.4  # 减小油门
            elevator = 0.08  # 轻微推杆下降
            if sim_time == 180:
                print("开始下降进近...")
        elif sim_time < 260:  # 220-260秒：稳定下降
            throttle = 0.3
            elevator = 0.06
            if sim_time == 220:
                print("稳定下降阶段...")
        else:  # 最后阶段：最终进近
            throttle = 0.2
            elevator = 0.04
            if sim_time == 260:
                print("最终进近阶段...")
        
        # 如果高度过低，停止仿真
        if current_alt < 150:
            print(f"着陆完成！最终高度: {current_alt:.0f}ft")
            break
        
        collector.set_controls(elevator=elevator, throttle=throttle)
        
        if not collector.fdm.run():
            break
        
        # 记录数据
        if i % 10 == 0:
            data = collector.get_flight_data()
            collector.data_log.append(data)
            
            if i % 200 == 0:  # 每2秒显示一次状态
                print(f"下降中 - 高度: {data['altitude_ft']:.0f}ft, 速度: {data['airspeed_kt']:.1f}kt, 下降率: {data['vertical_speed_fpm']:.0f}fpm")
    
    print("\n🎯 飞行周期完成！")
    print("=" * 30)
    
    # 保存数据和分析
    collector.save_data_to_files()
    
    # 飞行周期分析
    if collector.data_log:
        print("\n📊 完整飞行周期分析:")
        print("=" * 40)
        
        # 起飞分析
        takeoff_data = [d for d in collector.data_log if d['time'] <= 60]
        if takeoff_data:
            initial_alt = takeoff_data[0]['altitude_ft']
            takeoff_final_alt = takeoff_data[-1]['altitude_ft']
            max_takeoff_climb = max([d['vertical_speed_fpm'] for d in takeoff_data])
            print(f"🛫 起飞阶段:")
            print(f"   起始高度: {initial_alt:.0f}ft")
            print(f"   起飞后高度: {takeoff_final_alt:.0f}ft")
            print(f"   高度增益: {takeoff_final_alt - initial_alt:.0f}ft")
            print(f"   最大爬升率: {max_takeoff_climb:.0f}fpm")
        
        # 巡航分析
        cruise_data = [d for d in collector.data_log if 60 < d['time'] <= 180]
        if cruise_data:
            avg_cruise_alt = sum([d['altitude_ft'] for d in cruise_data]) / len(cruise_data)
            avg_cruise_speed = sum([d['airspeed_kt'] for d in cruise_data]) / len(cruise_data)
            print(f"\n✈️ 巡航阶段:")
            print(f"   平均高度: {avg_cruise_alt:.0f}ft")
            print(f"   平均速度: {avg_cruise_speed:.1f}kt")
        
        # 降落分析
        landing_data = [d for d in collector.data_log if d['time'] > 180]
        if landing_data:
            landing_start_alt = landing_data[0]['altitude_ft']
            final_alt = landing_data[-1]['altitude_ft']
            max_descent_rate = min([d['vertical_speed_fpm'] for d in landing_data])
            print(f"\n🛬 降落阶段:")
            print(f"   开始下降高度: {landing_start_alt:.0f}ft")
            print(f"   最终高度: {final_alt:.0f}ft")
            print(f"   高度损失: {landing_start_alt - final_alt:.0f}ft")
            print(f"   最大下降率: {max_descent_rate:.0f}fpm")
        
        # 整体统计
        all_altitudes = [d['altitude_ft'] for d in collector.data_log]
        all_speeds = [d['airspeed_kt'] for d in collector.data_log]
        max_altitude = max(all_altitudes)
        max_speed = max(all_speeds)
        flight_time = collector.data_log[-1]['time'] - collector.data_log[0]['time']
        
        print(f"\n📈 整体飞行统计:")
        print(f"   总飞行时间: {flight_time:.1f}秒")
        print(f"   最大高度: {max_altitude:.0f}ft")
        print(f"   最大速度: {max_speed:.1f}kt")
        print(f"   数据点数量: {len(collector.data_log)}")
        
        if final_alt < 200:
            print("\n✅ 成功完成完整飞行周期！")
        else:
            print("\n⚠️ 飞行周期部分完成")

if __name__ == "__main__":
    print("🚁 JSBSim 飞行数据采集器")
    print("=" * 50)
    
    try:
        # 检查JSBSim是否可用
        test_fdm = jsbsim.FGFDMExec(None)
        print("✅ JSBSim 库已正确安装")
        
        print("\n选择演示模式:")
        print("1. 基本飞行数据收集")
        print("2. 带控制输入的飞行")
        print("3. 完整飞行周期（起飞-飞行-降落）")
        
        choice = input("请选择 (1, 2 或 3): ").strip()
        
        if choice == "1":
            demo_basic_flight()
        elif choice == "2":
            demo_controlled_flight()
        elif choice == "3":
            demo_complete_flight_cycle()
        else:
            print("❌ 无效选择")
            
    except ImportError:
        print("❌ JSBSim 库未安装")
        print("💡 安装方法: pip install jsbsim")
    except Exception as e:
        print(f"❌ 错误: {e}")