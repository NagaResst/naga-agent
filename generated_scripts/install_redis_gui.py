#!/usr/bin/env python3
"""
Another Redis Desktop Manager 一键安装助手
会自动检测系统并下载对应版本，提示用户手动安装
"""

import os
import platform
import urllib.request
import sys
from pathlib import Path

def get_system_info():
    """获取系统信息"""
    system = platform.system()
    machine = platform.machine()
    
    if system == "Windows":
        return "windows", "x64"
    elif system == "Darwin":  # macOS
        return "macos", "arm64" if machine == "arm64" else "x64"
    elif system == "Linux":
        return "linux", "x64"
    else:
        return None, None

def get_download_url(system, arch):
    """根据系统获取下载地址"""
    base_url = "https://github.com/qishibo/AnotherRedisDesktopManager/releases/download/v1.7.1/"
    
    if system == "windows":
        return f"{base_url}Another.Redis.Desktop.Manager.v1.7.1.exe"
    elif system == "macos":
        return f"{base_url}Another.Redis.Desktop.Manager.v1.7.1.dmg"
    elif system == "linux":
        return f"{base_url}Another-Redis-Desktop-Manager-v1.7.1.AppImage"
    else:
        return None

def download_file(url, save_path):
    """下载文件"""
    print(f"正在下载：{url}")
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"下载完成！保存到：{save_path}")
        return True
    except Exception as e:
        print(f"下载失败：{e}")
        return False

def main():
    print("🔍 正在检测您的系统...")
    system, arch = get_system_info()
    
    if not system:
        print("❌ 不支持的系统类型，请手动下载安装")
        print("下载地址：https://github.com/qishibo/AnotherRedisDesktopManager/releases")
        return
    
    print(f"✅ 检测到系统：{system} ({arch})")
    
    url = get_download_url(system, arch)
    if not url:
        print("❌ 无法获取下载链接")
        return
    
    # 确定保存路径
    downloads_dir = Path.home() / "Downloads"
    downloads_dir.mkdir(exist_ok=True)
    
    filename = url.split("/")[-1]
    save_path = downloads_dir / filename
    
    # 检查是否已存在
    if save_path.exists():
        print(f"⚠️  文件已存在：{save_path}")
        overwrite = input("是否重新下载？(y/n): ").lower()
        if overwrite != 'y':
            print("使用现有文件")
            goto_install(save_path, system)
            return
    
    # 下载
    if download_file(url, str(save_path)):
        goto_install(save_path, system)

def goto_install(file_path, system):
    """引导用户安装"""
    print("\n" + "="*50)
    print("🎉 下载完成！请按以下步骤安装：")
    print("="*50)
    
    if system == "windows":
        print(f"1. 打开文件夹：{file_path.parent}")
        print(f"2. 双击运行：{file_path.name}")
        print("3. 一路点击'下一步'即可完成安装")
    elif system == "macos":
        print(f"1. 打开文件夹：{file_path.parent}")
        print(f"2. 双击打开：{file_path.name}")
        print("3. 将图标拖到 Applications 文件夹")
    elif system == "linux":
        print(f"1. 打开终端")
        print(f"2. 运行命令:")
        print(f"   chmod +x {file_path}")
        print(f"   {file_path}")
        print("或者用软件中心打开 AppImage 文件")
    
    print("\n💡 安装完成后，打开软件连接本地 Redis:")
    print("   Host: localhost")
    print("   Port: 6379")
    print("   Password: (如果没设密码就留空)")
    print("\n有任何问题随时问我哦～")

if __name__ == "__main__":
    main()