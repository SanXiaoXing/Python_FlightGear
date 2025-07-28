# Install

```shell
pip install --no-index --find-links=wheels/ $(ls wheels/*.whl)
```

```
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
```

