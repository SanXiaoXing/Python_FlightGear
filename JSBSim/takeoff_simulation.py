#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
飞机起飞仿真脚本
使用JSBSim仿真飞机起飞过程，输出关键飞行数据
"""

import jsbsim
import time
import csv
import os
from datetime import datetime

class TakeoffSimulation:
    def __init__(self, aircraft_model="c172x"):
        """
        初始化起飞仿真
        
        Args:
            aircraft_model: 飞机模型名称
        """
        self.fdm = jsbsim.FGFDMExec(None)
        self.aircraft_model = aircraft_model
        self.data_log = []
        
    def initialize_aircraft(self):
        """
        初始化飞机和起飞条件
        """
        try:
            # 加载飞机模型
            if not self.fdm.load_model(self.aircraft_model):
                raise Exception(f"无法加载飞机模型: {self.aircraft_model}")
            
            # 设置时间步长
            self.fdm.set_dt(0.01)  # 10ms步长
            
            # 设置起飞初始条件（地面）
            initial_conditions = {
                'ic/h-sl-ft': 100,         # 海拔高度100英尺
                'ic/h-agl-ft': 0,          # 距地高度0英尺（地面）
                'ic/long-gc-deg': -122.0,  # 经度
                'ic/lat-gc-deg': 37.0,     # 纬度
                'ic/u-fps': 0,             # 初始前向速度0（静止）
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
                raise Exception("飞机初始化失败")
            
            print(f"✅ 飞机 {self.aircraft_model} 初始化成功，准备起飞")
            return True
            
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            return False
    
    def get_flight_data(self):
        """
        获取当前飞行数据
        
        Returns:
            dict: 包含关键飞行参数的字典
        """
        return {
            # 时间
            'time_sec': self.fdm.get_sim_time(),
            
            # 高度数据
            'altitude_agl_ft': self.fdm.get_property_value('position/h-agl-ft'),  # 距地高度
            'altitude_msl_ft': self.fdm.get_property_value('position/h-sl-ft'),   # 海拔高度
            
            # 速度数据
            'airspeed_kt': self.fdm.get_property_value('velocities/vc-kts'),      # 指示空速
            'groundspeed_kt': self.fdm.get_property_value('velocities/vg-fps') * 0.592484,  # 地速
            'vertical_speed_fpm': self.fdm.get_property_value('velocities/h-dot-fps') * 60,  # 垂直速度
            
            # 姿态角数据
            'pitch_deg': self.fdm.get_property_value('attitude/pitch-rad') * 57.2958,  # 俯仰角
            'roll_deg': self.fdm.get_property_value('attitude/roll-rad') * 57.2958,    # 滚转角
            'yaw_deg': self.fdm.get_property_value('attitude/heading-true-rad') * 57.2958,  # 偏航角
            
            # 控制面和油门
            'throttle': self.fdm.get_property_value('fcs/throttle-pos-norm'),     # 油门位置
            'elevator': self.fdm.get_property_value('fcs/elevator-pos-norm'),     # 升降舵
            'aileron': self.fdm.get_property_value('fcs/aileron-pos-norm'),       # 副翼
            'rudder': self.fdm.get_property_value('fcs/rudder-pos-norm'),         # 方向舵
            
            # 发动机数据
            'engine_rpm': self.fdm.get_property_value('propulsion/engine/engine-rpm'),
            'fuel_flow': self.fdm.get_property_value('propulsion/engine/fuel-flow-rate-pps'),
        }
    
    def set_controls(self, throttle=None, elevator=None, aileron=None, rudder=None):
        """
        设置飞机控制输入
        
        Args:
            throttle: 油门位置 (0-1)
            elevator: 升降舵位置 (-1到1)
            aileron: 副翼位置 (-1到1)
            rudder: 方向舵位置 (-1到1)
        """
        if throttle is not None:
            self.fdm.set_property_value('fcs/throttle-cmd-norm', throttle)
        if elevator is not None:
            self.fdm.set_property_value('fcs/elevator-cmd-norm', elevator)
        if aileron is not None:
            self.fdm.set_property_value('fcs/aileron-cmd-norm', aileron)
        if rudder is not None:
            self.fdm.set_property_value('fcs/rudder-cmd-norm', rudder)
    
    def simulate_takeoff(self, duration_seconds=120):
        """
        仿真起飞过程
        
        Args:
            duration_seconds: 仿真持续时间（秒）
        """
        print("\n🛫 开始起飞仿真")
        print("=" * 50)
        
        # 起飞阶段划分
        phases = {
            'ground_roll': (0, 30),      # 地面滑跑阶段
            'rotation': (30, 45),        # 抬轮阶段
            'initial_climb': (45, 90),   # 初始爬升阶段
            'climb_out': (90, 120)       # 离场爬升阶段
        }
        
        current_phase = 'ground_roll'
        
        for i in range(duration_seconds * 100):  # 0.01秒步长
            sim_time = i * 0.01
            
            # 确定当前飞行阶段
            for phase, (start, end) in phases.items():
                if start <= sim_time < end:
                    if current_phase != phase:
                        current_phase = phase
                        print(f"\n📍 进入 {self.get_phase_name(phase)} 阶段")
                    break
            
            # 根据阶段设置控制输入
            if current_phase == 'ground_roll':
                # 地面滑跑：全油门，保持方向
                throttle = 1.0
                elevator = 0.0
                rudder = 0.0
                
            elif current_phase == 'rotation':
                # 抬轮：全油门，轻微拉杆
                throttle = 1.0
                elevator = -0.1  # 轻微拉杆
                rudder = 0.0
                
            elif current_phase == 'initial_climb':
                # 初始爬升：保持油门，调整俯仰
                throttle = 0.9
                elevator = -0.15  # 保持爬升姿态
                rudder = 0.0
                
            elif current_phase == 'climb_out':
                # 离场爬升：巡航油门，稳定爬升
                throttle = 0.8
                elevator = -0.1
                rudder = 0.0
            
            # 应用控制输入
            self.set_controls(throttle=throttle, elevator=elevator, rudder=rudder)
            
            # 运行仿真步骤
            if not self.fdm.run():
                print("❌ 仿真运行失败")
                break
            
            # 记录数据（每0.1秒记录一次）
            if i % 10 == 0:
                data = self.get_flight_data()
                self.data_log.append(data)
                
                # 实时显示关键数据
                if i % 100 == 0:  # 每秒显示一次
                    self.display_flight_status(data, current_phase)
                
                # 检查起飞成功条件
                if data['altitude_agl_ft'] > 50 and data['airspeed_kt'] > 60:
                    if current_phase in ['ground_roll', 'rotation']:
                        print(f"\n🎉 起飞成功！高度: {data['altitude_agl_ft']:.0f}ft, 速度: {data['airspeed_kt']:.1f}kt")
        
        print("\n✅ 起飞仿真完成")
        self.save_data()
        self.analyze_takeoff()
    
    def get_phase_name(self, phase):
        """获取阶段中文名称"""
        phase_names = {
            'ground_roll': '地面滑跑',
            'rotation': '抬轮',
            'initial_climb': '初始爬升',
            'climb_out': '离场爬升'
        }
        return phase_names.get(phase, phase)
    
    def display_flight_status(self, data, phase):
        """
        显示飞行状态
        
        Args:
            data: 飞行数据字典
            phase: 当前飞行阶段
        """
        print(f"[{self.get_phase_name(phase)}] "
              f"时间: {data['time_sec']:.1f}s | "
              f"距地高度: {data['altitude_agl_ft']:.1f}ft | "
              f"空速: {data['airspeed_kt']:.1f}kt | "
              f"油门: {data['throttle']:.2f} | "
              f"俯仰: {data['pitch_deg']:.1f}° | "
              f"滚转: {data['roll_deg']:.1f}°")
    
    def save_data(self):
        """
        保存飞行数据到CSV文件
        """
        if not self.data_log:
            print("❌ 没有数据可保存")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"takeoff_data_{timestamp}.csv"
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = self.data_log[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for data in self.data_log:
                    writer.writerow(data)
            
            print(f"📊 数据已保存到: {filename}")
            print(f"📈 共记录 {len(self.data_log)} 个数据点")
            
        except Exception as e:
            print(f"❌ 保存数据失败: {e}")
    
    def analyze_takeoff(self):
        """
        分析起飞性能
        """
        if not self.data_log:
            return
        
        print("\n📊 起飞性能分析")
        print("=" * 30)
        
        # 找到离地时刻
        liftoff_data = None
        for data in self.data_log:
            if data['altitude_agl_ft'] > 5:  # 离地5英尺
                liftoff_data = data
                break
        
        if liftoff_data:
            print(f"🛫 离地时间: {liftoff_data['time_sec']:.1f}秒")
            print(f"🛫 离地速度: {liftoff_data['airspeed_kt']:.1f}节")
            print(f"🛫 离地距离: 约{liftoff_data['time_sec'] * liftoff_data['groundspeed_kt'] * 0.514:.0f}米")
        
        # 最终状态
        final_data = self.data_log[-1]
        print(f"\n📈 最终状态:")
        print(f"   最大高度: {max([d['altitude_agl_ft'] for d in self.data_log]):.0f}英尺")
        print(f"   最大速度: {max([d['airspeed_kt'] for d in self.data_log]):.1f}节")
        print(f"   最大爬升率: {max([d['vertical_speed_fpm'] for d in self.data_log]):.0f}英尺/分钟")
        print(f"   最大俯仰角: {max([d['pitch_deg'] for d in self.data_log]):.1f}度")
        
        # 起飞成功判断
        if final_data['altitude_agl_ft'] > 100 and final_data['airspeed_kt'] > 70:
            print("\n✅ 起飞成功！")
        else:
            print("\n⚠️ 起飞未完全成功")

def main():
    """
    主函数
    """
    print("🛫 JSBSim 飞机起飞仿真")
    print("=" * 40)
    
    try:
        # 检查JSBSim是否可用
        test_fdm = jsbsim.FGFDMExec(None)
        print("✅ JSBSim 库已正确安装")
        
        # 创建起飞仿真实例
        simulation = TakeoffSimulation("c172x")
        
        # 初始化飞机
        if simulation.initialize_aircraft():
            # 开始起飞仿真
            simulation.simulate_takeoff(duration_seconds=120)
        else:
            print("❌ 飞机初始化失败")
            
    except ImportError:
        print("❌ JSBSim 库未安装")
        print("💡 安装方法: pip install jsbsim")
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    main()