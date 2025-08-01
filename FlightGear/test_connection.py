#!/usr/bin/env python3
"""
FlightGear连接测试脚本
使用flightgear-python库测试与FlightGear的Telnet连接

使用方法：
1. 启动FlightGear: fgfs --aircraft=c172p --airport=KSFO --telnet=socket,bi,60,localhost,5500,tcp
2. 运行此脚本: python test_connection.py
"""

import time
from flightgear_python.fg_if import TelnetConnection

def test_flightgear_connection():
    """测试FlightGear连接"""
    print("FlightGear连接测试")
    print("="*40)
    
    # 创建连接
    fg = TelnetConnection('localhost', 5500)
    
    try:
        print("正在连接到FlightGear...")
        fg.connect()
        print("✓ 连接成功！")
        
        # 测试基本属性读取
        print("\n测试属性读取:")
        aircraft = fg.get_prop('/sim/aircraft')
        print(f"✓ 当前飞机: {aircraft}")
        
        lat = fg.get_prop('/position/latitude-deg')
        lon = fg.get_prop('/position/longitude-deg')
        alt = fg.get_prop('/position/altitude-ft')
        print(f"✓ 当前位置: {lat:.6f}, {lon:.6f}")
        print(f"✓ 当前高度: {alt:.1f} 英尺")
        
        heading = fg.get_prop('/orientation/heading-deg')
        speed = fg.get_prop('/velocities/airspeed-kt')
        print(f"✓ 当前航向: {heading:.1f}°")
        print(f"✓ 当前空速: {speed:.1f} 节")
        
        # 测试属性设置
        print("\n测试属性设置:")
        original_heading = fg.get_prop('/autopilot/settings/heading-bug-deg')
        test_heading = 90.0
        
        fg.set_prop('/autopilot/settings/heading-bug-deg', test_heading)
        time.sleep(0.5)
        new_heading = fg.get_prop('/autopilot/settings/heading-bug-deg')
        
        if abs(new_heading - test_heading) < 1.0:
            print(f"✓ 属性设置成功: 航向设置为 {new_heading:.1f}°")
        else:
            print(f"✗ 属性设置失败: 期望 {test_heading}°, 实际 {new_heading:.1f}°")
        
        # 恢复原始值
        fg.set_prop('/autopilot/settings/heading-bug-deg', original_heading)
        
        print("\n✓ 所有测试通过！FlightGear连接正常工作。")
        
    except Exception as e:
        print(f"✗ 连接失败: {e}")
        print("\n请检查:")
        print("1. FlightGear是否正在运行")
        print("2. 启动参数是否正确: --telnet=socket,bi,60,localhost,5500,tcp")
        print("3. 端口5500是否被占用")
        return False
    
    finally:
        try:
            fg.close()
            print("连接已关闭")
        except:
            pass
    
    return True

if __name__ == '__main__':
    test_flightgear_connection()