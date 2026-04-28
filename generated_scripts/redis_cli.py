#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地 Redis 访问工具
支持基本的数据读写和查询操作
"""

import redis
import sys

class RedisCLI:
    def __init__(self, host='localhost', port=6379, db=0, password=None):
        self.client = redis.Redis(host=host, port=port, db=db, password=password)
    
    def test_connection(self):
        """测试连接是否成功"""
        try:
            return self.client.ping()
        except Exception as e:
            print(f"连接失败: {e}")
            return False
    
    def get(self, key):
        """获取键值"""
        value = self.client.get(key)
        if value is None:
            print(f"{key} 不存在")
        else:
            print(f"{key}: {value.decode('utf-8')}")
    
    def set(self, key, value, expire=None):
        """设置键值"""
        if expire:
            self.client.setex(key, expire, value)
            print(f"已设置 {key}, 过期时间: {expire}s")
        else:
            self.client.set(key, value)
            print(f"已设置 {key}")
    
    def delete(self, *keys):
        """删除键"""
        count = self.client.delete(*keys)
        print(f"已删除 {count} 个键")
    
    def keys(self, pattern='*'):
        """列出匹配的键"""
        result = self.client.keys(pattern)
        if not result:
            print("没有找到匹配的键")
        else:
            for key in result:
                print(key.decode('utf-8'))
    
    def type(self, key):
        """获取键的类型"""
        t = self.client.type(key).decode('utf-8')
        print(f"{key} 的类型: {t}")
    
    def ttl(self, key):
        """获取键的剩余生存时间"""
        t = self.client.ttl(key)
        if t == -1:
            print(f"{key} 没有设置过期时间")
        elif t == -2:
            print(f"{key} 不存在")
        else:
            print(f"{key} 剩余时间: {t}s")
    
    def hgetall(self, key):
        """获取哈希表所有字段"""
        data = self.client.hgetall(key)
        if not data:
            print(f"{key} 不存在")
        else:
            for k, v in data.items():
                print(f"  {k.decode('utf-8')}: {v.decode('utf-8')}")
    
    def lrange(self, key, start=0, end=-1):
        """获取列表范围"""
        data = self.client.lrange(key, start, end)
        if not data:
            print(f"{key} 不存在或为空")
        else:
            for i, item in enumerate(data):
                print(f"  [{i}] {item.decode('utf-8')}")


def main():
    print("=" * 50)
    print("      本地 Redis 访问工具")
    print("=" * 50)
    
    # 连接配置
    host = input("Redis 主机 (默认 localhost): ").strip() or "localhost"
    port = input("端口 (默认 6379): ").strip() or "6379"
    db = input("数据库 (默认 0): ").strip() or "0"
    password = input("密码 (无则回车): ").strip() or None
    
    cli = RedisCLI(host, int(port), int(db), password)
    
    if not cli.test_connection():
        print("连接失败，请检查配置")
        return
    
    print("\n连接成功！可用命令:")
    print("  get <key>           - 获取键值")
    print("  set <key> <value>   - 设置键值")
    print("  del <key> ...       - 删除键")
    print("  keys [pattern]      - 列出键")
    print("  type <key>          - 查看类型")
    print("  ttl <key>           - 查看过期时间")
    print("  hgetall <key>       - 查看哈希表")
    print("  lrange <key>        - 查看列表")
    print("  exit                - 退出")
    print("-" * 50)
    
    while True:
        try:
            cmd = input("\n> ").strip()
            if not cmd:
                continue
            if cmd.lower() == 'exit':
                print("再见～")
                break
            
            parts = cmd.split()
            action = parts[0].lower()
            
            if action == 'get' and len(parts) >= 2:
                cli.get(parts[1])
            elif action == 'set' and len(parts) >= 3:
                cli.set(parts[1], parts[2])
            elif action == 'del' and len(parts) >= 2:
                cli.delete(*parts[1:])
            elif action == 'keys':
                pattern = parts[1] if len(parts) > 1 else '*'
                cli.keys(pattern)
            elif action == 'type' and len(parts) >= 2:
                cli.type(parts[1])
            elif action == 'ttl' and len(parts) >= 2:
                cli.ttl(parts[1])
            elif action == 'hgetall' and len(parts) >= 2:
                cli.hgetall(parts[1])
            elif action == 'lrange' and len(parts) >= 2:
                cli.lrange(parts[1])
            else:
                print("未知命令，输入 exit 退出")
        except KeyboardInterrupt:
            print("\n再见～")
            break
        except Exception as e:
            print(f"错误: {e}")


if __name__ == '__main__':
    main()