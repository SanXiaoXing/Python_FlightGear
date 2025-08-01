import socket
import time
import math
import threading
from enum import Enum

class FlightPhase(Enum):
    PREFLIGHT = "preflight"
    TAXI = "taxi"
    TAKEOFF_ROLL = "takeoff_roll"
    ROTATION = "rotation"
    CLIMB = "climb"
    CRUISE = "cruise"
    DESCENT = "descent"
    APPROACH = "approach"
    LANDING = "landing"
    TAXI_IN = "taxi_in"

class IntelligentAutopilot:
    def __init__(self, host='localhost', telnet_port=5401, data_port=5500):
        self.host = host
        self.telnet_port = telnet_port
        self.data_port = data_port
        self.telnet_socket = None
        self.data_socket = None
        
        # 飞行状态
        self.current_phase = FlightPhase.PREFLIGHT
        self.flight_data = {}
        self.target_waypoint = None
        self.waypoints = []
        self.current_waypoint_index = 0
        
        # 飞行参数
        self.takeoff_speed = 65  # 起飞速度 (kt)
        self.rotation_speed = 55  # 拉起速度 (kt)
        self.climb_rate = 500    # 爬升率 (fpm)
        self.cruise_altitude = 3000
        self.cruise_speed = 120
        
        # 监控线程
        self.monitoring = False
        self.monitor_thread = None
        
    def connect(self):
        """连接到FlightGear"""
        try:
            # Telnet连接
            self.telnet_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.telnet_socket.connect((self.host, self.telnet_port))
            print("已连接到FlightGear telnet接口")
            
            # 数据接收
            self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.data_socket.bind(('', self.data_port))
            self.data_socket.settimeout(1.0)
            print("数据接收端口已设置")
            
            return True
        except Exception as e:
            print(f"连接失败: {e}")
            return False
            
    def send_command(self, command):
        """发送命令到FlightGear"""
        if self.telnet_socket:
            try:
                self.telnet_socket.send((command + '\n').encode())
                time.sleep(0.05)
            except Exception as e:
                print(f"命令发送失败: {e}")
                
    def set_property(self, property_path, value):
        """设置FlightGear属性"""
        command = f"set {property_path} {value}"
        self.send_command(command)
        
    def get_flight_data(self):
        """获取飞行数据"""
        try:
            data, addr = self.data_socket.recvfrom(1024)
            line = data.decode().strip()
            
            if line and not line.startswith('Altitude'):
                parts = line.split(',')
                if len(parts) >= 4:
                    self.flight_data = {
                        'altitude': float(parts[0]),
                        'latitude': float(parts[1]),
                        'longitude': float(parts[2]),
                        'pitch': float(parts[3]),
                        'airspeed': self.get_airspeed(),
                        'on_ground': self.is_on_ground()
                    }
                    return True
        except socket.timeout:
            pass
        except Exception as e:
            print(f"数据获取错误: {e}")
        return False
        
    def get_airspeed(self):
        """获取空速（模拟）"""
        # 实际应该从数据流中获取
        return 0  # 需要在XML配置中添加空速
        
    def is_on_ground(self):
        """检查是否在地面"""
        return self.flight_data.get('altitude', 0) < 50
        
    def calculate_distance_to_waypoint(self, lat, lon):
        """计算到航点的距离"""
        if not self.target_waypoint:
            return float('inf')
            
        target_lat, target_lon = self.target_waypoint
        
        # 简化的距离计算
        dlat = target_lat - lat
        dlon = target_lon - lon
        distance = math.sqrt(dlat*dlat + dlon*dlon) * 111000  # 转换为米
        
        return distance
        
    def ai_decision_engine(self):
        """AI决策引擎"""
        if not self.flight_data:
            return
            
        altitude = self.flight_data['altitude']
        airspeed = self.flight_data.get('airspeed', 0)
        on_ground = self.flight_data['on_ground']
        
        # 状态机逻辑
        if self.current_phase == FlightPhase.PREFLIGHT:
            self.execute_preflight()
            
        elif self.current_phase == FlightPhase.TAKEOFF_ROLL:
            self.execute_takeoff_roll(airspeed)
            
        elif self.current_phase == FlightPhase.ROTATION:
            self.execute_rotation(airspeed, altitude)
            
        elif self.current_phase == FlightPhase.CLIMB:
            self.execute_climb(altitude)
            
        elif self.current_phase == FlightPhase.CRUISE:
            self.execute_cruise()
            
        elif self.current_phase == FlightPhase.APPROACH:
            self.execute_approach(altitude)
            
        elif self.current_phase == FlightPhase.LANDING:
            self.execute_landing(altitude, on_ground)
            
    def execute_preflight(self):
        """执行起飞前检查"""
        print("执行起飞前检查...")
        
        # 设置起飞配置
        self.set_property("/controls/flight/flaps", 0.2)  # 起飞襟翼
        self.set_property("/controls/gear/brake-parking", "false")  # 释放停车刹车
        self.set_property("/controls/engines/engine[0]/mixture", 1.0)  # 混合比
        self.set_property("/controls/engines/engine[0]/propeller-pitch", 1.0)  # 螺旋桨桨距
        
        time.sleep(2)
        self.current_phase = FlightPhase.TAKEOFF_ROLL
        print("进入起飞滑跑阶段")
        
    def execute_takeoff_roll(self, airspeed):
        """执行起飞滑跑"""
        # 逐渐增加油门
        throttle = min(1.0, airspeed / self.takeoff_speed)
        self.set_property("/controls/engines/engine[0]/throttle", throttle)
        
        # 方向舵控制保持跑道中心线
        self.set_property("/controls/flight/rudder", 0)
        
        print(f"起飞滑跑中... 当前空速: {airspeed}kt")
        
        # 达到拉起速度
        if airspeed >= self.rotation_speed:
            self.current_phase = FlightPhase.ROTATION
            print("达到拉起速度，开始拉起")
            
    def execute_rotation(self, airspeed, altitude):
        """执行拉起"""
        # 轻柔拉起
        elevator = -0.3  # 负值为拉起
        self.set_property("/controls/flight/elevator", elevator)
        
        print(f"拉起中... 高度: {altitude}ft, 空速: {airspeed}kt")
        
        # 离地后进入爬升
        if altitude > 50:
            self.current_phase = FlightPhase.CLIMB
            print("已离地，进入爬升阶段")
            
    def execute_climb(self, altitude):
        """执行爬升"""
        # 收起起落架
        if altitude > 200:
            self.set_property("/controls/gear/gear-down", "false")
            
        # 收起襟翼
        if altitude > 500:
            self.set_property("/controls/flight/flaps", 0)
            
        # 爬升姿态控制
        target_pitch = 10  # 目标俯仰角
        elevator = -0.1 if self.flight_data.get('pitch', 0) < target_pitch else 0.1
        self.set_property("/controls/flight/elevator", elevator)
        
        print(f"爬升中... 当前高度: {altitude}ft")
        
        # 达到巡航高度
        if altitude >= self.cruise_altitude:
            self.current_phase = FlightPhase.CRUISE
            print(f"达到巡航高度 {self.cruise_altitude}ft")
            
    def execute_cruise(self):
        """执行巡航"""
        # 保持高度和航向
        self.set_property("/autopilot/locks/altitude", "altitude-hold")
        self.set_property("/autopilot/settings/target-altitude-ft", self.cruise_altitude)
        
        # 航向控制
        if self.target_waypoint:
            lat = self.flight_data['latitude']
            lon = self.flight_data['longitude']
            
            # 计算到航点的距离
            distance = self.calculate_distance_to_waypoint(lat, lon)
            
            if distance < 1000:  # 1km内认为到达航点
                self.next_waypoint()
                
        print(f"巡航中... 高度: {self.flight_data['altitude']}ft")
        
    def execute_approach(self, altitude):
        """执行进近"""
        # 下降到进近高度
        self.set_property("/controls/flight/flaps", 0.5)  # 进近襟翼
        
        # 放下起落架
        if altitude < 2000:
            self.set_property("/controls/gear/gear-down", "true")
            
        print(f"进近中... 高度: {altitude}ft")
        
        if altitude < 500:
            self.current_phase = FlightPhase.LANDING
            
    def execute_landing(self, altitude, on_ground):
        """执行降落"""
        # 全襟翼
        self.set_property("/controls/flight/flaps", 1.0)
        
        # 减小油门
        self.set_property("/controls/engines/engine[0]/throttle", 0.2)
        
        print(f"降落中... 高度: {altitude}ft")
        
        if on_ground:
            print("已着陆！")
            self.set_property("/controls/gear/brake-left", 1.0)
            self.set_property("/controls/gear/brake-right", 1.0)
            
    def set_flight_plan(self, waypoints):
        """设置飞行计划"""
        self.waypoints = waypoints
        self.current_waypoint_index = 0
        if waypoints:
            self.target_waypoint = waypoints[0]
            print(f"飞行计划已设置，共{len(waypoints)}个航点")
            
    def next_waypoint(self):
        """前往下一个航点"""
        self.current_waypoint_index += 1
        if self.current_waypoint_index < len(self.waypoints):
            self.target_waypoint = self.waypoints[self.current_waypoint_index]
            print(f"前往航点 {self.current_waypoint_index + 1}")
        else:
            print("所有航点已完成，开始进近")
            self.current_phase = FlightPhase.APPROACH
            
    def start_monitoring(self):
        """开始监控线程"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop)
        self.monitor_thread.start()
        
    def monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            if self.get_flight_data():
                self.ai_decision_engine()
            time.sleep(0.1)
            
    def execute_intelligent_flight(self, waypoints):
        """执行智能飞行"""
        print("开始智能飞行...")
        
        # 设置飞行计划
        self.set_flight_plan(waypoints)
        
        # 开始监控
        self.start_monitoring()
        
        try:
            while self.monitoring:
                time.sleep(1)
                
                # 检查是否完成飞行
                if self.current_phase == FlightPhase.TAXI_IN:
                    print("飞行完成！")
                    break
                    
        except KeyboardInterrupt:
            print("飞行中断")
        finally:
            self.monitoring = False
            
    def stop(self):
        """停止飞行"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        if self.telnet_socket:
            self.telnet_socket.close()
        if self.data_socket:
            self.data_socket.close()

# 使用示例
if __name__ == "__main__":
    # 创建智能自动驾驶
    autopilot = IntelligentAutopilot()
    
    if autopilot.connect():
        # 定义航点
        waypoints = [
            (37.621311, -122.378968),  # 起点
            (37.7749, -122.4194),     # 航点1
            (37.621311, -122.378968)   # 返回起点
        ]
        
        try:
            # 执行智能飞行
            autopilot.execute_intelligent_flight(waypoints)
        finally:
            autopilot.stop()