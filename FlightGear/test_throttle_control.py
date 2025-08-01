import telnetlib
import time
import socket

def test_flightgear_connection():
    """测试FlightGear连接和油门控制"""
    try:
        print("🔌 正在连接FlightGear...")
        # 连接FlightGear Telnet接口
        tn = telnetlib.Telnet("localhost", 5501, timeout=5)
        print("✅ 成功连接到FlightGear")
        
        # 测试多种油门控制路径
        throttle_paths = [
            "/controls/engines/engine[0]/throttle",  # 正确的路径
            "/controls/engines/engine/throttle",     # 旧的路径
            "/fdm/jsbsim/propulsion/engine[0]/throttle-cmd-norm"
        ]
        
        print("\n🧪 测试油门控制路径...")
        
        for i, path in enumerate(throttle_paths, 1):
            print(f"\n--- 测试路径 {i}: {path} ---")
            
            # 设置油门值
            test_value = 0.8
            set_cmd = f"set {path} {test_value}"
            tn.write(set_cmd.encode('ascii') + b"\n")
            time.sleep(1)
            
            # 读取油门值
            get_cmd = f"get {path}"
            tn.write(get_cmd.encode('ascii') + b"\n")
            time.sleep(0.5)
            
            # 尝试读取响应
            try:
                response = tn.read_very_eager().decode('ascii')
                print(f"设置值: {test_value}")
                print(f"响应: {response.strip()}")
                
                # 检查是否设置成功
                if str(test_value) in response or abs(float(response.split()[-1]) - test_value) < 0.1:
                    print(f"✅ 路径 {path} 工作正常")
                else:
                    print(f"❌ 路径 {path} 可能无效")
            except Exception as e:
                print(f"❌ 读取响应失败: {e}")
        
        # 设置自动驾驶仪参数（使用正确的油门路径）
        print("\n🛩️ 设置自动驾驶仪参数...")
        commands = [
            "set /autopilot/settings/target-speed-kt 150",
            "set /autopilot/settings/target-altitude-ft 6000",
            "set /controls/engines/engine[0]/throttle 0.8",  # 使用正确的路径
            "set /autopilot/locks/heading heading-hold",
            "set /autopilot/locks/altitude altitude-hold",
            "set /autopilot/locks/speed speed-with-throttle"
        ]
        
        for cmd in commands:
            print(f"执行: {cmd}")
            tn.write(cmd.encode('ascii') + b"\n")
            time.sleep(0.5)
        
        print("\n✅ 自动驾驶仪设置完成")
        
        # 检查当前状态
        print("\n📊 检查当前飞机状态...")
        status_commands = [
            "get /controls/engines/engine[0]/throttle",
            "get /velocities/airspeed-kt",
            "get /position/altitude-ft",
            "get /orientation/heading-deg"
        ]
        
        for cmd in status_commands:
            tn.write(cmd.encode('ascii') + b"\n")
            time.sleep(0.5)
            try:
                response = tn.read_very_eager().decode('ascii')
                print(f"{cmd}: {response.strip()}")
            except:
                print(f"{cmd}: 无法读取")
        
        # 询问用户是否继续监控
        print("\n🔄 是否要持续监控飞行状态？(y/n): ", end="")
        choice = input().lower()
        
        if choice == 'y':
            print("\n📡 开始监控飞行状态 (按Ctrl+C停止)...")
            try:
                while True:
                    # 每10秒检查一次状态
                    time.sleep(10)
                    tn.write(b"get /velocities/airspeed-kt\n")
                    time.sleep(0.2)
                    speed_response = tn.read_very_eager().decode('ascii')
                    
                    tn.write(b"get /position/altitude-ft\n")
                    time.sleep(0.2)
                    alt_response = tn.read_very_eager().decode('ascii')
                    
                    print(f"速度: {speed_response.strip()} kt, 高度: {alt_response.strip()} ft")
            except KeyboardInterrupt:
                print("\n⏹️ 监控已停止")
        
        tn.close()
        print("\n🔌 连接已关闭")
        
    except socket.timeout:
        print("❌ 连接超时 - 请确保FlightGear已启动并启用Telnet")
    except ConnectionRefusedError:
        print("❌ 连接被拒绝 - 请检查FlightGear是否在端口5501上监听")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        print("\n💡 解决方案:")
        print("1. 确保FlightGear已启动")
        print("2. 启动时添加参数: --telnet=5501")
        print("3. 检查防火墙设置")

if __name__ == "__main__":
    print("🚁 FlightGear 油门控制测试工具")
    print("=" * 40)
    test_flightgear_connection()