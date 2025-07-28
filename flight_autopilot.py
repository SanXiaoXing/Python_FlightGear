import socket
import time
import math

class FlightGearAutopilot:
    def __init__(self, host='localhost', telnet_port=5401, data_port=5500):
        self.host = host
        self.telnet_port = telnet_port
        self.data_port = data_port
        self.telnet_socket = None
        self.data_socket = None
        
    def connect(self):
        """连接到FlightGear"""
        try:
            # 连接到FlightGear的telnet接口
            self.telnet_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.telnet_socket.connect((self.host, self.telnet_port))
            print("已连接到FlightGear telnet接口")
            
            # 设置数据接收
            self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.data_socket.bind(('', self.data_port))
            print("数据接收端口已设置")
            
        except Exception as e:
            print(f"连接失败: {e}")
            
    def send_command(self, command):
        """发送命令到FlightGear"""
        if self.telnet_socket:
            try:
                self.telnet_socket.send((command + '\n').encode())
                time.sleep(0.1)
            except Exception as e:
                print(f"命令发送失败: {e}")
                
    def set_property(self, property_path, value):
        """设置FlightGear属性"""
        command = f"set {property_path} {value}"
        self.send_command(command)
        
    def get_property(self, property_path):
        """获取FlightGear属性"""
        command = f"get {property_path}"
        self.send_command(command)
        
    def set_position(self, latitude, longitude, altitude=1000):
        """设置飞机位置"""
        self.set_property("/position/latitude-deg", latitude)
        self.set_property("/position/longitude-deg", longitude)
        self.set_property("/position/altitude-ft", altitude)
        print(f"位置设置为: {latitude}, {longitude}, {altitude}ft")
        
    def enable_autopilot(self):
        """启用自动驾驶"""
        # 启用自动驾驶主开关
        self.set_property("/autopilot/locks/passive-mode", "false")
        
        # 启用高度保持
        self.set_property("/autopilot/locks/altitude", "altitude-hold")
        
        # 启用航向保持
        self.set_property("/autopilot/locks/heading", "dg-heading-hold")
        
        # 启用速度保持
        self.set_property("/autopilot/locks/speed", "speed-with-throttle")
        
        print("自动驾驶已启用")
        
    def set_autopilot_targets(self, heading, altitude, speed):
        """设置自动驾驶目标"""
        self.set_property("/autopilot/settings/target-altitude-ft", altitude)
        self.set_property("/autopilot/settings/heading-bug-deg", heading)
        self.set_property("/autopilot/settings/target-speed-kt", speed)
        print(f"自动驾驶目标: 航向{heading}°, 高度{altitude}ft, 速度{speed}kt")
        
    def calculate_heading(self, lat1, lon1, lat2, lon2):
        """计算两点间的航向"""
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        dlon = lon2 - lon1
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        
        heading = math.atan2(y, x)
        heading = math.degrees(heading)
        heading = (heading + 360) % 360
        
        return heading
        
    def auto_takeoff(self, runway_heading, target_altitude=3000):
        """自动起飞"""
        print("开始自动起飞程序...")
        
        # 设置起飞配置
        self.set_property("/controls/flight/flaps", 0.3)  # 襟翼
        self.set_property("/controls/engines/engine[0]/throttle", 1.0)  # 全油门
        self.set_property("/controls/gear/brake-parking", "false")  # 释放停车刹车
        
        # 设置航向
        self.set_property("/autopilot/settings/heading-bug-deg", runway_heading)
        
        # 等待速度建立
        time.sleep(5)
        
        # 启用自动驾驶
        self.enable_autopilot()
        
        # 设置爬升目标
        self.set_autopilot_targets(runway_heading, target_altitude, 150)
        
        print(f"起飞完成，爬升至{target_altitude}ft")
        
    def auto_landing(self, airport_lat, airport_lon, runway_heading):
        """自动降落"""
        print("开始自动降落程序...")
        
        # 设置进近配置
        self.set_property("/controls/flight/flaps", 0.8)  # 降落襟翼
        
        # 设置降落航向
        self.set_autopilot_targets(runway_heading, 1000, 120)
        
        # 启用ILS进近（如果可用）
        self.set_property("/autopilot/locks/altitude", "gs1-hold")
        self.set_property("/autopilot/locks/heading", "nav1-hold")
        
        print("降落程序已启动")
        
    def execute_flight_plan(self, waypoints, cruise_altitude=10000, cruise_speed=200):
        """执行飞行计划"""
        print("开始执行飞行计划...")
        
        for i, waypoint in enumerate(waypoints):
            lat, lon = waypoint
            print(f"飞向航点 {i+1}: {lat}, {lon}")
            
            if i == 0:
                # 第一个航点，起飞
                heading = self.calculate_heading(
                    waypoints[0][0], waypoints[0][1],
                    waypoints[1][0], waypoints[1][1]
                )
                self.auto_takeoff(heading, cruise_altitude)
            elif i == len(waypoints) - 1:
                # 最后一个航点，降落
                heading = self.calculate_heading(
                    waypoints[i-1][0], waypoints[i-1][1],
                    lat, lon
                )
                self.auto_landing(lat, lon, heading)
            else:
                # 中间航点，巡航
                heading = self.calculate_heading(
                    waypoints[i-1][0], waypoints[i-1][1],
                    lat, lon
                )
                self.set_autopilot_targets(heading, cruise_altitude, cruise_speed)
                
            # 等待到达航点（简化版本）
            time.sleep(30)  # 实际应该监控位置
            
    def monitor_flight(self):
        """监控飞行状态"""
        try:
            while True:
                data, addr = self.data_socket.recvfrom(1024)
                flight_data = data.decode().strip()
                print(f"飞行数据: {flight_data}")
                time.sleep(1)
        except KeyboardInterrupt:
            print("监控停止")
            
    def disconnect(self):
        """断开连接"""
        if self.telnet_socket:
            self.telnet_socket.close()
        if self.data_socket:
            self.data_socket.close()
        print("连接已断开")

# 使用示例
if __name__ == "__main__":
    # 创建自动驾驶实例
    autopilot = FlightGearAutopilot()
    
    # 连接到FlightGear
    autopilot.connect()
    
    # 定义飞行计划（纬度，经度）
    flight_plan = [
        (37.621311, -122.378968),  # 旧金山机场
        (37.7749, -122.4194),     # 旧金山市区
        (37.8044, -122.2711),     # 奥克兰
        (37.6213, -122.3790)      # 返回起点
    ]
    
    try:
        # 设置起始位置
        autopilot.set_position(flight_plan[0][0], flight_plan[0][1], 100)
        
        # 等待位置设置
        time.sleep(2)
        
        # 执行飞行计划
        autopilot.execute_flight_plan(flight_plan, cruise_altitude=8000, cruise_speed=180)
        
        # 监控飞行（可选）
        # autopilot.monitor_flight()
        
    except KeyboardInterrupt:
        print("飞行中断")
    finally:
        autopilot.disconnect()