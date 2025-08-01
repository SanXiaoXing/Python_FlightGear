""" 
    测试使用指令进行飞行功能测试
"""
from flightgear_python.fg_if import TelnetConnection
import time


class TestFlightGear:
    def __init__(self, host='localhost', port=5500):
        """初始化FlightGear Telnet连接"""
        self.fg = TelnetConnection(host, port)
        self.connected = False
        

    def connect(self):
        """连接到FlightGear"""
        try:
            print(f"正在连接到FlightGear Telnet服务器: {self.fg.host}:{self.fg.port}")
            self.fg.connect()
            test_prop = self.fg.get_prop('/sim/aircraft')
            altitude = autopilot.fg.get_prop('/position/altitude-ft')
            airspeed = autopilot.fg.get_prop('/velocities/airspeed-kt')
            
            print(f"📊 当前状态:")
            print(f"   高度: {altitude:.1f} 英尺")
            print(f"   空速: {airspeed:.1f} 节")
            self.connected = True
            print(f"成功连接到FlightGear！当前飞机: {test_prop}")
            return True
        except Exception as e:
            print(f"连接FlightGear失败: {e}")
            print("请确保FlightGear正在运行并启用了Telnet服务器")
            self.connected = False
            return False
        
    def set_flight(self):
        """ 
        设置飞机参数
        """
        try:
            self.connect()
            if not self.connected:
                print("\n无法连接到FlightGear，请检查：")
                print("1. FlightGear是否正在运行")
                print("2. 是否启用了Telnet服务器 (--telnet=socket,bi,60,localhost,5500,tcp)")
                print("3. 端口5500是否被占用")
                exit(1)
            # 使用正确的油门控制路径（包含数组索引）
            self.fg.set_prop('/controls/engines/engine[0]/throttle', 1.0)
            time.sleep(1)  # 等待设置生效
            throtte = self.fg.get_prop('/controls/engines/engine[0]/throttle')
            
            # 同时检查其他可能的油门路径
            alt_paths = [
                '/controls/engines/engine/throttle',
                '/fdm/jsbsim/propulsion/engine[0]/throttle-cmd-norm'
            ]
            
            print(f"主要油门路径值: {throtte}")
            for path in alt_paths:
                try:
                    value = self.fg.get_prop(path)
                    print(f"备用路径 {path}: {value}")
                except:
                    print(f"备用路径 {path}: 无法读取")
            print('-'*20)
            print(throtte)
            print('-'*20)


        except Exception as e:
            print(f"\n程序运行出错: {e}")


if __name__ == "__main__":

    autopilot = TestFlightGear()
    try:
        autopilot.set_flight()
        if not autopilot.connected:
            print("\n无法连接到FlightGear，请检查：")
            print("1. FlightGear是否正在运行")
            print("2. 是否启用了Telnet服务器 (--telnet=socket,bi,60,localhost,5500,tcp)")
            print("3. 端口5500是否被占用")
            exit(1)

    except Exception as e:
        print(f"\n程序运行出错: {e}")
        