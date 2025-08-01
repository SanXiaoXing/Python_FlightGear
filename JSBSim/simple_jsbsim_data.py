#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版 JSBSim 飞行数据获取器
专门用于获取飞机飞行过程中的关键数据
"""

import jsbsim
import time
import json
import pandas as pd
from datetime import datetime

def get_flight_data_simple():
    """
    简单的JSBSim飞行数据获取示例
    """
    print("🛩️ 初始化 JSBSim...")
    
    # 创建JSBSim实例
    fdm = jsbsim.FGFDMExec(None)
    
    # 加载飞机模型（Cessna 172）
    if not fdm.load_model('c172x'):
        print("❌ 无法加载飞机模型")
        return None
    
    # 设置初始条件
    fdm.set_property_value('ic/h-sl-ft', 5000)      # 高度5000英尺
    fdm.set_property_value('ic/u-fps', 120)         # 前向速度120英尺/秒
    fdm.set_property_value('ic/lat-gc-deg', 37.0)   # 纬度
    fdm.set_property_value('ic/long-gc-deg', -122.0) # 经度
    
    # 初始化
    if not fdm.run_ic():
        print("❌ 初始化失败")
        return None
    
    print("✅ JSBSim 初始化成功")
    
    # 数据收集
    flight_data = []
    
    print("📊 开始收集飞行数据...")
    
    # 运行仿真并收集数据
    for i in range(300):  # 收集300个数据点（约30秒）
        # 运行一步仿真
        if not fdm.run():
            break
        
        # 获取关键飞行数据
        data_point = {
            '时间(秒)': fdm.get_sim_time(),
            '纬度(度)': fdm.get_property_value('position/lat-gc-deg'),
            '经度(度)': fdm.get_property_value('position/long-gc-deg'),
            '高度(英尺)': fdm.get_property_value('position/h-sl-ft'),
            '空速(节)': fdm.get_property_value('velocities/vc-kts'),
            '地速(节)': fdm.get_property_value('velocities/vg-fps') * 0.592484,
            '垂直速度(英尺/分)': fdm.get_property_value('velocities/h-dot-fps') * 60,
            '滚转角(度)': fdm.get_property_value('attitude/phi-deg'),
            '俯仰角(度)': fdm.get_property_value('attitude/theta-deg'),
            '航向角(度)': fdm.get_property_value('attitude/psi-deg'),
            '油门位置': fdm.get_property_value('fcs/throttle-pos-norm'),
            '升降舵位置': fdm.get_property_value('fcs/elevator-pos-norm'),
            '发动机转速(RPM)': fdm.get_property_value('propulsion/engine/engine-rpm'),
        }
        
        flight_data.append(data_point)
        
        # 每50个数据点显示一次进度
        if i % 50 == 0:
            print(f"时间: {data_point['时间(秒)']:.1f}s, 高度: {data_point['高度(英尺)']:.0f}ft, 速度: {data_point['空速(节)']:.1f}kt")
    
    print(f"✅ 数据收集完成，共 {len(flight_data)} 个数据点")
    return flight_data

def save_data(data, format='both'):
    """
    保存飞行数据
    
    Args:
        data: 飞行数据列表
        format: 保存格式 ('json', 'csv', 'both')
    """
    if not data:
        print("❌ 没有数据可保存")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format in ['json', 'both']:
        # 保存为JSON格式
        json_file = f"jsbsim_flight_data_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"💾 JSON数据已保存: {json_file}")
    
    if format in ['csv', 'both']:
        # 保存为CSV格式（使用pandas）
        try:
            df = pd.DataFrame(data)
            csv_file = f"jsbsim_flight_data_{timestamp}.csv"
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            print(f"📊 CSV数据已保存: {csv_file}")
        except ImportError:
            print("⚠️ pandas未安装，跳过CSV保存")

def analyze_flight_data(data):
    """
    分析飞行数据
    
    Args:
        data: 飞行数据列表
    """
    if not data:
        print("❌ 没有数据可分析")
        return
    
    print("\n📈 飞行数据分析")
    print("=" * 40)
    
    # 提取关键数据
    altitudes = [d['高度(英尺)'] for d in data]
    airspeeds = [d['空速(节)'] for d in data]
    times = [d['时间(秒)'] for d in data]
    
    print(f"飞行时长: {max(times):.1f} 秒")
    print(f"最大高度: {max(altitudes):.0f} 英尺")
    print(f"最小高度: {min(altitudes):.0f} 英尺")
    print(f"平均高度: {sum(altitudes)/len(altitudes):.0f} 英尺")
    print(f"最大空速: {max(airspeeds):.1f} 节")
    print(f"最小空速: {min(airspeeds):.1f} 节")
    print(f"平均空速: {sum(airspeeds)/len(airspeeds):.1f} 节")
    
    # 计算高度变化
    if len(altitudes) > 1:
        alt_change = altitudes[-1] - altitudes[0]
        print(f"总高度变化: {alt_change:.0f} 英尺")

def get_real_time_data(duration_seconds=10):
    """
    实时获取JSBSim数据
    
    Args:
        duration_seconds: 数据获取持续时间
    """
    print(f"🔄 实时数据获取 ({duration_seconds}秒)")
    
    fdm = jsbsim.FGFDMExec(None)
    
    if not fdm.load_model('c172x'):
        print("❌ 无法加载飞机模型")
        return
    
    # 设置初始条件
    fdm.set_property_value('ic/h-sl-ft', 3000)
    fdm.set_property_value('ic/u-fps', 100)
    fdm.run_ic()
    
    start_time = time.time()
    
    try:
        while (time.time() - start_time) < duration_seconds:
            fdm.run()
            
            # 获取实时数据
            current_data = {
                '时间': fdm.get_sim_time(),
                '高度': fdm.get_property_value('position/h-sl-ft'),
                '空速': fdm.get_property_value('velocities/vc-kts'),
                '航向': fdm.get_property_value('attitude/psi-deg'),
            }
            
            print(f"\r时间: {current_data['时间']:.1f}s | 高度: {current_data['高度']:.0f}ft | 空速: {current_data['空速']:.1f}kt | 航向: {current_data['航向']:.1f}°", end="")
            
            time.sleep(0.1)  # 100ms更新间隔
            
    except KeyboardInterrupt:
        print("\n⏹️ 实时数据获取已停止")
    
    print("\n✅ 实时数据获取完成")

def main():
    """
    主函数 - 提供用户选择界面
    """
    print("🚁 JSBSim 飞行数据获取工具")
    print("=" * 50)
    
    try:
        # 检查JSBSim是否可用
        test_fdm = jsbsim.FGFDMExec(None)
        print("✅ JSBSim 库检测成功")
        
        while True:
            print("\n📋 选择功能:")
            print("1. 获取完整飞行数据")
            print("2. 实时数据监控")
            print("3. 退出")
            
            choice = input("\n请选择 (1-3): ").strip()
            
            if choice == '1':
                print("\n🛩️ 开始获取完整飞行数据...")
                flight_data = get_flight_data_simple()
                
                if flight_data:
                    # 分析数据
                    analyze_flight_data(flight_data)
                    
                    # 询问是否保存
                    save_choice = input("\n💾 是否保存数据? (y/n): ").lower()
                    if save_choice == 'y':
                        format_choice = input("保存格式 (json/csv/both): ").lower()
                        if format_choice not in ['json', 'csv', 'both']:
                            format_choice = 'both'
                        save_data(flight_data, format_choice)
            
            elif choice == '2':
                duration = input("\n⏱️ 监控时长(秒，默认10): ").strip()
                try:
                    duration = int(duration) if duration else 10
                except ValueError:
                    duration = 10
                
                get_real_time_data(duration)
            
            elif choice == '3':
                print("👋 再见！")
                break
            
            else:
                print("❌ 无效选择，请重试")
    
    except ImportError:
        print("❌ JSBSim 库未安装")
        print("💡 安装命令: pip install jsbsim")
        print("📖 或者: conda install -c conda-forge jsbsim")
    
    except Exception as e:
        print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    main()