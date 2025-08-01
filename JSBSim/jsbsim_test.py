#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSBSim 属性浏览器
用于查找和浏览 JSBSim 中所有可用的属性参数
"""

import jsbsim
import json
from collections import defaultdict

class JSBSimPropertyBrowser:
    def __init__(self, aircraft_model="c172x"):
        self.fdm = jsbsim.FGFDMExec(None)
        self.aircraft_model = aircraft_model
        self.properties = []
        
    def initialize(self):
        """初始化 JSBSim 并加载飞机模型"""
        try:
            if not self.fdm.load_model(self.aircraft_model):
                raise Exception(f"无法加载飞机模型: {self.aircraft_model}")
            
            if not self.fdm.run_ic():
                raise Exception("初始化失败")
            
            print(f"✅ 成功加载飞机模型: {self.aircraft_model}")
            return True
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            return False
    
    def get_all_properties(self):
        """获取所有可用属性"""
        self.properties = self.fdm.query_property_catalog('')
        print(f"📊 发现 {len(self.properties)} 个可用属性")
        return self.properties
    
    def categorize_properties(self):
        """按类别分组属性"""
        categories = defaultdict(list)
        
        for prop in self.properties:
            # 按第一级路径分类
            if '/' in prop:
                category = prop.split('/')[0]
                categories[category].append(prop)
            else:
                categories['root'].append(prop)
        
        return dict(categories)
    
    def search_properties(self, keyword):
        """搜索包含关键词的属性"""
        keyword = keyword.lower()
        matching = [p for p in self.properties if keyword in p.lower()]
        return matching
    
    def get_property_value(self, property_name):
        """获取属性值"""
        try:
            value = self.fdm.get_property_value(property_name)
            return value
        except:
            return None
    
    def display_categories(self):
        """显示所有类别"""
        categories = self.categorize_properties()
        
        print("\n📋 属性类别概览:")
        print("=" * 50)
        
        for category, props in sorted(categories.items()):
            print(f"\n🔹 {category.upper()} ({len(props)} 个属性)")
            
            # 显示前5个属性作为示例
            for prop in props[:5]:
                value = self.get_property_value(prop)
                if value is not None:
                    print(f"  ├─ {prop}: {value}")
                else:
                    print(f"  ├─ {prop}: [无法获取值]")
            
            if len(props) > 5:
                print(f"  └─ ... 还有 {len(props) - 5} 个属性")
    
    def display_category_details(self, category_name):
        """显示特定类别的详细信息"""
        categories = self.categorize_properties()
        
        if category_name not in categories:
            print(f"❌ 未找到类别: {category_name}")
            print(f"可用类别: {', '.join(categories.keys())}")
            return
        
        props = categories[category_name]
        print(f"\n📋 {category_name.upper()} 类别详细信息 ({len(props)} 个属性):")
        print("=" * 60)
        
        for i, prop in enumerate(props, 1):
            value = self.get_property_value(prop)
            if value is not None:
                print(f"{i:3d}. {prop:<40} = {value}")
            else:
                print(f"{i:3d}. {prop:<40} = [无法获取值]")
    
    def search_and_display(self, keyword):
        """搜索并显示匹配的属性"""
        matches = self.search_properties(keyword)
        
        if not matches:
            print(f"❌ 未找到包含 '{keyword}' 的属性")
            return
        
        print(f"\n🔍 搜索结果: '{keyword}' ({len(matches)} 个匹配):")
        print("=" * 60)
        
        for i, prop in enumerate(matches, 1):
            value = self.get_property_value(prop)
            if value is not None:
                print(f"{i:3d}. {prop:<40} = {value}")
            else:
                print(f"{i:3d}. {prop:<40} = [无法获取值]")
    
    def save_properties_to_file(self, filename="jsbsim_properties.json"):
        """保存所有属性到文件"""
        categories = self.categorize_properties()
        
        # 创建包含属性值的数据结构
        data = {
            'aircraft_model': self.aircraft_model,
            'total_properties': len(self.properties),
            'categories': {}
        }
        
        for category, props in categories.items():
            data['categories'][category] = {
                'count': len(props),
                'properties': {}
            }
            
            for prop in props:
                value = self.get_property_value(prop)
                data['categories'][category]['properties'][prop] = value
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"📄 属性数据已保存到: {filename}")
        except Exception as e:
            print(f"❌ 保存失败: {e}")

def main():
    """主函数 - 交互式属性浏览器"""
    print("🔍 JSBSim 属性浏览器")
    print("=" * 40)
    
    browser = JSBSimPropertyBrowser()
    
    if not browser.initialize():
        return
    
    browser.get_all_properties()
    
    while True:
        print("\n📋 选择操作:")
        print("1. 显示所有类别概览")
        print("2. 查看特定类别详情")
        print("3. 搜索属性")
        print("4. 保存属性到文件")
        print("5. 退出")
        
        choice = input("\n请选择 (1-5): ").strip()
        
        if choice == '1':
            browser.display_categories()
        
        elif choice == '2':
            categories = browser.categorize_properties()
            print(f"\n可用类别: {', '.join(categories.keys())}")
            category = input("请输入类别名称: ").strip()
            browser.display_category_details(category)
        
        elif choice == '3':
            keyword = input("请输入搜索关键词: ").strip()
            if keyword:
                browser.search_and_display(keyword)
        
        elif choice == '4':
            filename = input("请输入文件名 (默认: jsbsim_properties.json): ").strip()
            if not filename:
                filename = "jsbsim_properties.json"
            browser.save_properties_to_file(filename)
        
        elif choice == '5':
            print("👋 再见！")
            break
        
        else:
            print("❌ 无效选择，请重试")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，再见！")
    except Exception as e:
        print(f"\n❌ 程序错误: {e}")