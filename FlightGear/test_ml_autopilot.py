#!/usr/bin/env python3
"""
FlightGear机器学习自动驾驶系统 - 快速测试脚本
用于验证系统功能和连接性
"""

import time
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ml_autopilot import MLFlightGearAutopilot, FlightState
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保ml_autopilot.py文件在同一目录下")
    input("按回车键退出...")
    sys.exit(1)

def test_connection():
    """测试FlightGear连接"""
    print("🔗 测试FlightGear连接...")
    
    autopilot = MLFlightGearAutopilot()
    
    if autopilot.connect():
        print("✅ 连接成功！")
        
        # 测试基本属性读取
        try:
            aircraft = autopilot.fg.get_prop('/sim/aircraft')
            altitude = autopilot.fg.get_prop('/position/altitude-ft')
            airspeed = autopilot.fg.get_prop('/velocities/airspeed-kt')
            
            print(f"📊 当前状态:")
            print(f"   飞机型号: {aircraft}")
            print(f"   高度: {altitude:.1f} 英尺")
            print(f"   空速: {airspeed:.1f} 节")
            
            autopilot.stop()
            return True
            
        except Exception as e:
            print(f"❌ 读取飞机状态失败: {e}")
            autopilot.stop()
            return False
    else:
        print("❌ 连接失败")
        return False

def test_flight_state():
    """测试飞行状态读取"""
    print("\n📊 测试飞行状态读取...")
    
    autopilot = MLFlightGearAutopilot()
    
    if not autopilot.connect():
        print("❌ 无法连接到FlightGear")
        return False
    
    try:
        state = FlightState()
        if state.from_fg(autopilot.fg):
            print("✅ 飞行状态读取成功！")
            print(f"   高度: {state.altitude:.1f} ft")
            print(f"   空速: {state.airspeed:.1f} kt")
            print(f"   航向: {state.heading:.1f}°")
            print(f"   油门: {state.throttle:.2f}")
            print(f"   起落架: {'下' if state.gear_down else '上'}")
            print(f"   襟翼: {state.flaps:.2f}")
            print(f"   俯仰角: {state.pitch:.1f}°")
            print(f"   滚转角: {state.roll:.1f}°")
            print(f"   垂直速度: {state.vertical_speed:.1f} fpm")
            print(f"   地速: {state.ground_speed:.1f} kt")
            
            # 测试状态数组转换
            state_array = state.to_array()
            print(f"   状态数组长度: {len(state_array)}")
            print(f"   状态数组: {state_array}")
            
            autopilot.stop()
            return True
        else:
            print("❌ 飞行状态读取失败")
            autopilot.stop()
            return False
            
    except Exception as e:
        print(f"❌ 测试飞行状态时出错: {e}")
        autopilot.stop()
        return False

def test_basic_controls():
    """测试基本控制功能"""
    print("\n🎛️ 测试基本控制功能...")
    
    autopilot = MLFlightGearAutopilot()
    
    if not autopilot.connect():
        print("❌ 无法连接到FlightGear")
        return False
    
    try:
        print("📝 保存当前状态...")
        # 保存当前状态
        original_throttle = autopilot.fg.get_prop('/controls/engines/engine[0]/throttle')
        original_flaps = autopilot.fg.get_prop('/controls/flight/flaps')
        original_gear = autopilot.fg.get_prop('/controls/gear/gear-down')
        
        print(f"   原始油门: {original_throttle:.2f}")
        print(f"   原始襟翼: {original_flaps:.2f}")
        print(f"   原始起落架: {'下' if original_gear else '上'}")
        
        print("\n🧪 测试控制命令...")
        
        # 测试油门控制
        print("   测试油门控制...")
        autopilot.fg.set_prop('/controls/engines/engine[0]/throttle', 0.5)
        time.sleep(1)
        new_throttle = autopilot.fg.get_prop('/controls/engines/engine[0]/throttle')
        print(f"   设置油门0.5，实际: {new_throttle:.2f}")
        
        # 测试襟翼控制
        print("   测试襟翼控制...")
        autopilot.fg.set_prop('/controls/flight/flaps', 0.3)
        time.sleep(1)
        new_flaps = autopilot.fg.get_prop('/controls/flight/flaps')
        print(f"   设置襟翼0.3，实际: {new_flaps:.2f}")
        
        # 测试起落架控制
        print("   测试起落架控制...")
        autopilot.fg.set_prop('/controls/gear/gear-down', True)
        time.sleep(1)
        new_gear = autopilot.fg.get_prop('/controls/gear/gear-down')
        print(f"   设置起落架放下，实际: {'下' if new_gear else '上'}")
        
        print("\n🔄 恢复原始状态...")
        # 恢复原始状态
        autopilot.fg.set_prop('/controls/engines/engine/throttle', original_throttle)
        autopilot.fg.set_prop('/controls/flight/flaps', original_flaps)
        autopilot.fg.set_prop('/controls/gear/gear-down', original_gear)
        
        time.sleep(1)
        print("✅ 基本控制测试完成！")
        
        autopilot.stop()
        return True
        
    except Exception as e:
        print(f"❌ 测试基本控制时出错: {e}")
        autopilot.stop()
        return False

def test_takeoff_agent():
    """测试起飞代理"""
    print("\n🤖 测试起飞代理...")
    
    try:
        from ml_autopilot import TakeoffAgent, FlightState
        
        agent = TakeoffAgent()
        print("✅ 起飞代理创建成功！")
        
        # 创建测试状态
        test_state = FlightState()
        test_state.altitude = 100
        test_state.airspeed = 45
        test_state.throttle = 0.6
        test_state.gear_down = True
        test_state.flaps = 0.3
        
        print(f"📊 测试状态:")
        print(f"   高度: {test_state.altitude} ft")
        print(f"   空速: {test_state.airspeed} kt")
        print(f"   油门: {test_state.throttle}")
        
        # 测试阶段判断
        phase = agent.get_takeoff_phase(test_state)
        print(f"   判断阶段: {phase}")
        
        # 测试专家动作
        expert_action = agent.get_expert_action(test_state)
        print(f"   专家建议:")
        print(f"     油门变化: {expert_action.throttle_delta:.3f}")
        print(f"     起落架动作: {expert_action.gear_action}")
        print(f"     襟翼变化: {expert_action.flaps_delta:.3f}")
        
        # 测试AI动作选择
        ai_action = agent.act(test_state)
        print(f"   AI选择:")
        print(f"     油门变化: {ai_action.throttle_delta:.3f}")
        print(f"     起落架动作: {ai_action.gear_action}")
        print(f"     襟翼变化: {ai_action.flaps_delta:.3f}")
        
        print("✅ 起飞代理测试完成！")
        return True
        
    except Exception as e:
        print(f"❌ 测试起飞代理时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("FlightGear机器学习自动驾驶系统 - 快速测试")
    print("="*60)
    
    print("\n🚀 开始系统测试...")
    
    # 测试结果
    results = {
        '连接测试': False,
        '状态读取': False,
        '基本控制': False,
        '起飞代理': False
    }
    
    # 执行测试
    try:
        results['连接测试'] = test_connection()
        
        if results['连接测试']:
            results['状态读取'] = test_flight_state()
            results['基本控制'] = test_basic_controls()
        
        results['起飞代理'] = test_takeoff_agent()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 显示测试结果
    print("\n" + "="*60)
    print("📋 测试结果汇总:")
    print("-"*30)
    
    all_passed = True
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
        if not result:
            all_passed = False
    
    print("-"*30)
    
    if all_passed:
        print("🎉 所有测试通过！系统运行正常")
        print("\n📝 下一步:")
        print("1. 确保FlightGear在跑道上")
        print("2. 运行: python ml_autopilot.py")
        print("3. 选择测试模式进行起飞")
    else:
        print("⚠️ 部分测试失败，请检查:")
        if not results['连接测试']:
            print("   - FlightGear是否运行")
            print("   - Telnet服务器是否启用")
        if not results['状态读取'] or not results['基本控制']:
            print("   - FlightGear版本兼容性")
            print("   - 网络连接稳定性")
        if not results['起飞代理']:
            print("   - Python依赖包安装")
            print("   - 机器学习模块")
    
    print("\n" + "="*60)
    input("\n按回车键退出...")

if __name__ == '__main__':
    main()