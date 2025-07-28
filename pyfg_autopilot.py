""" 
FlightGear真实飞行控制系统 - 通过Telnet属性接口控制FlightGear中的飞机
使用flightgear-python官方库进行连接和控制
""" 
import time 
import math
from flightgear_python.fg_if import TelnetConnection
from tqdm import tqdm

class FlightGearRealAutopilot:
    """真实的FlightGear飞机控制系统"""
    
    def __init__(self, host='localhost', port=5500):
        """初始化FlightGear Telnet连接"""
        self.fg = TelnetConnection(host, port)
        self.connected = False
        
        # 飞行状态
        self.flight_phase = 'ground'  # ground, takeoff, climb, cruise, descent, approach, landing
        self.departure_point = None  # 起点 (lat, lon, alt)
        self.destination_point = None  # 终点 (lat, lon, alt)
        self.cruise_altitude = 3000  # 巡航高度（英尺）
        self.target_heading = 0
        
        # 飞行计划
        self.flight_plan = {
            'takeoff_altitude': 500,  # 起飞完成高度（英尺）
            'cruise_altitude': 10000,  # 巡航高度（英尺）
            'approach_altitude': 1000,  # 进近高度（英尺）
            'climb_rate': 1000,  # 爬升率 ft/min
            'descent_rate': 500,  # 下降率 ft/min
            'cruise_speed': 250,  # 巡航速度 knots
        } 

    def connect(self):
        """连接到FlightGear"""
        try:
            print(f"正在连接到FlightGear Telnet服务器: {self.fg.host}:{self.fg.port}")
            # 建立连接
            self.fg.connect()
            # 测试连接
            test_prop = self.fg.get_prop('/sim/aircraft')
            self.connected = True
            print(f"成功连接到FlightGear！当前飞机: {test_prop}")
            
            # 获取当前飞机状态
            self._get_current_status()
            
        except Exception as e:
            print(f"连接FlightGear失败: {e}")
            print("请确保FlightGear正在运行并启用了Telnet服务器")
            print("启动参数示例: --telnet=socket,bi,60,localhost,5500,tcp")
            self.connected = False
            
    def _get_current_status(self):
        """获取当前飞机状态"""
        if not self.connected:
            return
            
        try:
            lat = self.fg.get_prop('/position/latitude-deg')
            lon = self.fg.get_prop('/position/longitude-deg')
            alt = self.fg.get_prop('/position/altitude-ft')
            heading = self.fg.get_prop('/orientation/heading-deg')
            speed = self.fg.get_prop('/velocities/airspeed-kt')
            
            print(f"当前位置: {lat:.6f}, {lon:.6f}")
            print(f"当前高度: {alt:.1f} 英尺")
            print(f"当前航向: {heading:.1f}°")
            print(f"当前空速: {speed:.1f} 节")
            
        except Exception as e:
            print(f"获取飞机状态失败: {e}")
            
    def set_autopilot_properties(self, enable=True):
        """设置自动驾驶属性"""
        if not self.connected:
            print("未连接到FlightGear")
            return False
            
        try:
            # 启用自动驾驶
            self.fg.set_prop('/autopilot/locks/autopilot', 'enabled' if enable else '')
            
            # 设置自动驾驶模式
            if enable:
                self.fg.set_prop('/autopilot/locks/altitude', 'altitude-hold')
                self.fg.set_prop('/autopilot/locks/heading', 'dg-heading-hold')
                self.fg.set_prop('/autopilot/locks/speed', 'speed-with-throttle')
                print("自动驾驶已启用")
            else:
                self.fg.set_prop('/autopilot/locks/altitude', '')
                self.fg.set_prop('/autopilot/locks/heading', '')
                self.fg.set_prop('/autopilot/locks/speed', '')
                print("自动驾驶已禁用")
                
            return True
            
        except Exception as e:
            print(f"设置自动驾驶失败: {e}")
            return False
            
    def set_target_altitude(self, altitude_ft):
        """设置目标高度（英尺）"""
        if not self.connected:
            return False
            
        try:
            self.fg.set_prop('/autopilot/settings/target-altitude-ft', altitude_ft)
            print(f"目标高度设置为: {altitude_ft} 英尺")
            return True
        except Exception as e:
            print(f"设置目标高度失败: {e}")
            return False
    
    def set_target_heading(self, heading_deg):
        """设置目标航向（度）"""
        if not self.connected:
            return False
            
        try:
            self.fg.set_prop('/autopilot/settings/heading-bug-deg', heading_deg)
            print(f"目标航向设置为: {heading_deg}°")
            return True
        except Exception as e:
            print(f"设置目标航向失败: {e}")
            return False
    
    def set_target_speed(self, speed_kt):
        """设置目标空速（节）"""
        if not self.connected:
            return False
            
        try:
            self.fg.set_prop('/autopilot/settings/target-speed-kt', speed_kt)
            print(f"目标空速设置为: {speed_kt} 节")
            return True
        except Exception as e:
            print(f"设置目标空速失败: {e}")
            return False
            
    def set_position(self, lat, lon, alt_ft, heading=0):
        """设置飞机位置"""
        if not self.connected:
            return False
            
        try:
            self.fg.set_prop('/position/latitude-deg', lat)
            self.fg.set_prop('/position/longitude-deg', lon)
            self.fg.set_prop('/position/altitude-ft', alt_ft)
            self.fg.set_prop('/orientation/heading-deg', heading)
            print(f"飞机位置设置为: {lat:.6f}, {lon:.6f}, {alt_ft} 英尺, 航向 {heading}°")
            return True
        except Exception as e:
            print(f"设置飞机位置失败: {e}")
            return False
            
    def get_current_position(self):
        """获取当前位置"""
        if not self.connected:
            return None
            
        try:
            lat = self.fg.get_prop('/position/latitude-deg')
            lon = self.fg.get_prop('/position/longitude-deg')
            alt = self.fg.get_prop('/position/altitude-ft')
            heading = self.fg.get_prop('/orientation/heading-deg')
            return lat, lon, alt, heading
        except Exception as e:
            print(f"获取当前位置失败: {e}")
            return None
            
    def get_current_altitude(self):
        """获取当前高度（英尺）"""
        if not self.connected:
            return 0
            
        try:
            return self.fg.get_prop('/position/altitude-ft')
        except Exception as e:
            print(f"获取当前高度失败: {e}")
            return 0
        
    def set_flight_plan(self, departure_lat, departure_lon, departure_alt, 
                        destination_lat, destination_lon, destination_alt, cruise_alt=10000):
        """设置飞行计划"""
        self.departure_point = (departure_lat, departure_lon, departure_alt)
        self.destination_point = (destination_lat, destination_lon, destination_alt)
        self.cruise_altitude = cruise_alt
        
        # 计算航向
        self.target_heading = self.calculate_heading(
            departure_lat, departure_lon, destination_lat, destination_lon
        )
        
        distance = self.calculate_distance(
            departure_lat, departure_lon, destination_lat, destination_lon
        )
        
        print(f"飞行计划已设置:")
        print(f"起点: {departure_lat:.6f}, {departure_lon:.6f}, {departure_alt} 英尺")
        print(f"终点: {destination_lat:.6f}, {destination_lon:.6f}, {destination_alt} 英尺")
        print(f"航向: {self.target_heading:.1f}°")
        print(f"距离: {distance:.2f}km")
        print(f"巡航高度: {cruise_alt} 英尺")
        
        # 更新飞行计划参数
        self.flight_plan['cruise_altitude'] = cruise_alt
        
    def execute_complete_flight(self):
        """执行完整的飞行过程"""
        if not self.departure_point or not self.destination_point:
            print("错误：请先设置飞行计划")
            return
            
        if not self.connected:
            print("错误：未连接到FlightGear，请先调用connect()")
            return
            
        print("\n开始执行完整飞行过程...")
        print("="*50)
        
        try:
            # 0. 初始化飞机位置和状态
            self._initialize_flight()
            
            # 1. 起飞阶段
            self._execute_takeoff()
            
            # 2. 爬升阶段
            self._execute_climb()
            
            # 3. 巡航阶段
            self._execute_cruise()
            
            # 4. 下降阶段
            self._execute_descent()
            
            # 5. 进近和着陆
            self._execute_approach_and_landing()
            
            print("\n完整飞行过程执行完毕！")
            
        except KeyboardInterrupt:
            print("\n飞行过程被用户中断")
        except Exception as e:
            print(f"飞行过程中出现错误: {e}")
            
    def _initialize_flight(self):
        """初始化飞行状态"""
        print("\n阶段0: 初始化飞行状态")
        
        # 设置飞机到起始位置
        departure = self.departure_point
        self.set_position(departure[0], departure[1], departure[2], self.target_heading)
        
        # 设置初始速度和引擎状态
        try:
            self.fg.set_prop('/controls/engines/engine[0]/throttle', 0.8)  # 设置油门
            self.fg.set_prop('/controls/gear/gear-down', False)  # 收起起落架
            self.fg.set_prop('/velocities/airspeed-kt', 150)  # 设置初始空速
            print("飞机初始化完成")
        except Exception as e:
            print(f"初始化飞机状态失败: {e}")
            
    def _execute_takeoff(self):
        """执行起飞阶段"""
        print("\n阶段1: 起飞")
        self.flight_phase = 'takeoff'
        
        # 计算起飞目标高度
        target_alt = self.departure_point[2] + self.flight_plan['takeoff_altitude']
        
        # 启用自动驾驶并设置起飞参数
        self.set_autopilot_properties(True)
        self.set_target_altitude(target_alt)
        self.set_target_heading(self.target_heading)
        self.set_target_speed(180)  # 起飞速度
        
        print(f"起飞目标高度: {target_alt} 英尺")
        
        # 获取起始高度
        start_alt = self.get_current_altitude()
        total_climb = target_alt - start_alt
        
        # 使用tqdm显示起飞进度
        with tqdm(total=int(total_climb), desc="起飞进度", unit="英尺", 
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.0f}/{total:.0f} {unit} [{elapsed}<{remaining}]") as pbar:
            
            last_alt = start_alt
            
            # 监控起飞过程
            while True:
                current_alt = self.get_current_altitude()
                
                # 更新进度条
                progress = current_alt - start_alt
                if progress > last_alt - start_alt:
                    pbar.update(int(progress - (last_alt - start_alt)))
                    last_alt = current_alt
                
                # 设置进度条描述信息
                pbar.set_description(f"起飞进度 (当前: {current_alt:.0f}英尺)")
                
                if current_alt >= target_alt:
                    # 确保进度条达到100%
                    pbar.update(int(total_climb - pbar.n))
                    break
                    
                time.sleep(2.0)
        
        print("起飞完成！")
        
    def _execute_climb(self):
        """执行爬升阶段"""
        print("\n阶段2: 爬升到巡航高度")
        self.flight_phase = 'climb'
        
        # 设置巡航高度和速度
        self.set_target_altitude(self.cruise_altitude)
        self.set_target_speed(self.flight_plan['cruise_speed'])
        
        print(f"爬升目标高度: {self.cruise_altitude} 英尺")
        
        # 获取起始高度
        start_alt = self.get_current_altitude()
        total_climb = self.cruise_altitude - start_alt
        
        # 使用tqdm显示爬升进度
        with tqdm(total=int(total_climb), desc="爬升进度", unit="英尺",
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.0f}/{total:.0f} {unit} [{elapsed}<{remaining}]") as pbar:
            
            last_alt = start_alt
            
            # 监控爬升过程
            while True:
                current_alt = self.get_current_altitude()
                
                # 更新进度条
                progress = current_alt - start_alt
                if progress > last_alt - start_alt:
                    pbar.update(int(progress - (last_alt - start_alt)))
                    last_alt = current_alt
                
                # 设置进度条描述信息
                pbar.set_description(f"爬升进度 (当前: {current_alt:.0f}英尺)")
                
                if abs(current_alt - self.cruise_altitude) < 100:  # 允许100英尺误差
                    # 确保进度条达到100%
                    pbar.update(int(total_climb - pbar.n))
                    break
                    
                time.sleep(3.0)
        
        print("到达巡航高度！")
        
    def _execute_cruise(self):
        """执行巡航阶段"""
        print("\n阶段3: 巡航飞行")
        self.flight_phase = 'cruise'
        
        # 计算巡航时间（根据距离和速度）
        distance = self.calculate_distance(
            self.departure_point[0], self.departure_point[1],
            self.destination_point[0], self.destination_point[1]
        )
        
        # 估算巡航时间（距离km / 速度kt * 1.852 * 60分钟）
        cruise_time_minutes = max(5, int(distance / (self.flight_plan['cruise_speed'] * 1.852) * 60))
        cruise_steps = cruise_time_minutes  # 每分钟检查一次
        
        print(f"预计巡航时间: {cruise_time_minutes} 分钟")
        print(f"巡航距离: {distance:.1f} km")
        
        # 保持巡航状态
        for i in range(cruise_steps):
            current_alt = self.get_current_altitude()
            current_pos = self.get_current_position()
            
            if current_pos:
                lat, lon, alt, heading = current_pos
                remaining_distance = self.calculate_distance(
                    lat, lon, self.destination_point[0], self.destination_point[1]
                )
                
                progress = max(0, (distance - remaining_distance) / distance * 100)
                print(f"巡航中... 进度: {progress:.1f}%, 高度: {current_alt:.1f} 英尺, 剩余距离: {remaining_distance:.1f} km")
                
                # 如果接近目的地，提前结束巡航
                if remaining_distance < distance * 0.3:  # 剩余30%距离时开始下降
                    print("接近目的地，准备开始下降")
                    break
            else:
                print(f"巡航中... 步骤 {i+1}/{cruise_steps}")
                
            time.sleep(60)  # 每分钟检查一次
            
        print("巡航阶段完成！")
        
    def _execute_descent(self):
        """执行下降阶段"""
        print("\n阶段4: 下降")
        self.flight_phase = 'descent'
        
        target_alt = self.flight_plan['approach_altitude']
        
        # 设置下降目标高度和速度
        self.set_target_altitude(target_alt)
        self.set_target_speed(200)  # 下降速度
        
        print(f"下降目标高度: {target_alt} 英尺")
        
        # 监控下降过程
        while True:
            current_alt = self.get_current_altitude()
            if abs(current_alt - target_alt) < 100:  # 允许100英尺误差
                break
                
            print(f"下降中... 当前高度: {current_alt:.1f} 英尺 / 目标: {target_alt:.1f} 英尺")
            time.sleep(3.0)
            
        print("下降到进近高度！")
        
    def _execute_approach_and_landing(self):
        """执行进近和着陆阶段"""
        print("\n阶段5: 进近和着陆")
        self.flight_phase = 'approach'
        
        target_alt = self.destination_point[2]
        
        # 设置进近参数
        self.set_target_altitude(target_alt)
        self.set_target_speed(150)  # 进近速度
        
        # 计算到目的地的航向
        current_pos = self.get_current_position()
        if current_pos:
            lat, lon, alt, heading = current_pos
            final_heading = self.calculate_heading(
                lat, lon, self.destination_point[0], self.destination_point[1]
            )
            self.set_target_heading(final_heading)
            print(f"设置最终进近航向: {final_heading:.1f}°")
        
        print(f"进近目标高度: {target_alt} 英尺")
        
        # 监控进近过程
        while True:
            current_alt = self.get_current_altitude()
            current_pos = self.get_current_position()
            
            if current_pos:
                lat, lon, alt, heading = current_pos
                distance_to_dest = self.calculate_distance(
                    lat, lon, self.destination_point[0], self.destination_point[1]
                )
                
                print(f"进近中... 高度: {current_alt:.1f} 英尺, 距离目的地: {distance_to_dest:.2f} km")
                
                # 如果接近目的地且高度合适，准备着陆
                if distance_to_dest < 5 and abs(current_alt - target_alt) < 200:
                    break
            else:
                print(f"进近中... 当前高度: {current_alt:.1f} 英尺")
                if abs(current_alt - target_alt) < 100:
                    break
                    
            time.sleep(5.0)
            
        # 最终着陆阶段
        print("\n执行最终着陆...")
        try:
            self.fg.set_prop('/controls/gear/gear-down', True)  # 放下起落架
            self.fg.set_prop('/controls/flight/flaps', 1.0)     # 放下襟翼
            self.set_target_speed(120)  # 着陆速度
            time.sleep(3)
        except Exception as e:
            print(f"设置着陆配置失败: {e}")
            
        self.flight_phase = 'landing'
        print("着陆成功！")
        
    @staticmethod
    def calculate_heading(lat1, lon1, lat2, lon2):
        """计算两点间的航向"""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon_rad = math.radians(lon2 - lon1)
        
        y = math.sin(dlon_rad) * math.cos(lat2_rad)
        x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad)
        
        heading = math.degrees(math.atan2(y, x))
        return (heading + 360) % 360
        
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
        """断开FlightGear连接"""
        if self.connected:
            print("正在断开FlightGear连接...")
            try:
                # 禁用自动驾驶
                self.set_autopilot_properties(False)
                # 断开连接
                self.fg.close()
                self.connected = False
                print("FlightGear连接已断开")
            except Exception as e:
                print(f"断开连接时出错: {e}")
        else:
            print("未连接到FlightGear")

# 预设飞行路线（使用B和P开头的机场，高度单位：英尺）
FLIGHT_ROUTES = {
    'BIKF到BGBW': {
        'departure': (64.1300, -21.9406, 171),  # 雷克雅未克凯夫拉维克机场 BIKF
        'destination': (61.1572, -45.4258, 283), # 格陵兰纳萨尔苏阿克机场 BGBW
        'cruise_alt': 25000  # 巡航高度25000英尺
    },
    'PANC到PAFA': {
        'departure': (61.1744, -149.9961, 152),  # 阿拉斯加安克雷奇机场 PANC
        'destination': (64.8378, -147.8564, 434),   # 阿拉斯加费尔班克斯机场 PAFA
        'cruise_alt': 20000  # 巡航高度20000英尺
    },
    'PHNL到PHOG': {
        'departure': (21.3187, -157.9224, 13),  # 夏威夷檀香山机场 PHNL
        'destination': (20.8987, -156.4306, 80), # 夏威夷卡胡卢伊机场 PHOG
        'cruise_alt': 15000  # 巡航高度15000英尺
    },
    'BIRK到BIAR': {
        'departure': (64.1300, -21.9406, 171),  # 冰岛雷克雅未克机场 BIRK
        'destination': (65.6600, -18.0728, 18), # 冰岛阿克雷里机场 BIAR
        'cruise_alt': 18000  # 巡航高度18000英尺
    },
    'PAJN到PAKT': {
        'departure': (58.3548, -134.5763, 21),  # 阿拉斯加朱诺机场 PAJN
        'destination': (55.7556, -131.7139, 35), # 阿拉斯加凯奇坎机场 PAKT
        'cruise_alt': 16000  # 巡航高度16000英尺
    }
}

def get_user_flight_choice():
    """获取用户选择的飞行路线"""
    print("\n可选飞行路线:")
    routes = list(FLIGHT_ROUTES.keys())
    
    for i, route in enumerate(routes, 1):
        route_info = FLIGHT_ROUTES[route]
        print(f"{i}. {route}")
        print(f"   起点: {route_info['departure'][0]:.4f}, {route_info['departure'][1]:.4f}")
        print(f"   终点: {route_info['destination'][0]:.4f}, {route_info['destination'][1]:.4f}")
        print(f"   巡航高度: {route_info['cruise_alt']} 英尺")
        print()
    
    while True:
        try:
            choice = input(f"请选择飞行路线 (1-{len(routes)}): ").strip()
            if choice.isdigit():
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(routes):
                    selected_route = routes[choice_idx]
                    return selected_route, FLIGHT_ROUTES[selected_route]
            print("无效选择，请重新输入")
        except (ValueError, KeyboardInterrupt):
            print("\n程序退出")
            return None, None

""" 
启动FlightGear时使用以下参数:
--telnet=socket,bi,60,localhost,5500,tcp

完整启动命令示例:
fgfs --aircraft=c172p --airport=KSFO --telnet=socket,bi,60,localhost,5500,tcp

注意：
1. 确保FlightGear已启动并加载了飞机
2. 确保Telnet服务器端口5500可用
3. 建议使用C172或类似的通用航空飞机进行测试
""" 
if __name__ == '__main__':  # NOTE: This is REQUIRED on Windows! 
    print("FlightGear真实飞行控制系统")
    print("="*60)
    print("本系统通过Telnet接口控制FlightGear中的真实飞机")
    print("包括：起飞 -> 爬升 -> 巡航 -> 下降 -> 进近 -> 着陆")
    print("="*60)
    print("\n重要提示：")
    print("1. 请确保FlightGear已启动并启用Telnet服务器")
    print("2. 启动参数: --telnet=socket,bi,60,localhost,5500,tcp")
    print("3. 建议使用C172或类似飞机进行测试")
    print("="*60)
    
    # 创建飞行控制器实例
    autopilot = FlightGearRealAutopilot()
    
    try:
        # 连接到FlightGear
        autopilot.connect()
        
        if not autopilot.connected:
            print("\n无法连接到FlightGear，请检查：")
            print("1. FlightGear是否正在运行")
            print("2. 是否启用了Telnet服务器 (--telnet=socket,bi,60,localhost,5500,tcp)")
            print("3. 端口5500是否被占用")
            exit(1)
        
        # 获取用户选择的飞行路线
        route_name, route_info = get_user_flight_choice()
        
        if not route_info:
            print("未选择飞行路线，程序退出")
            exit(0)
        
        print(f"\n已选择飞行路线: {route_name}")
        
        # 设置飞行计划
        departure = route_info['departure']
        destination = route_info['destination']
        cruise_alt = route_info['cruise_alt']
        
        autopilot.set_flight_plan(
            departure[0], departure[1], departure[2],
            destination[0], destination[1], destination[2],
            cruise_alt
        )
        
        # 确认开始飞行
        input("\n按回车键开始执行完整飞行过程...")
        
        # 执行完整飞行
        autopilot.execute_complete_flight()
        
        print("\n" + "="*60)
        print("飞行任务完成！")
        print(f"成功完成 {route_name} 航线飞行")
        print("感谢使用FlightGear真实飞行控制系统！")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n飞行被用户中断")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        autopilot.stop()
        print("\n程序结束，再见！")