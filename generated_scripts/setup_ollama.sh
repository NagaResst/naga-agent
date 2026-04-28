#!/bin/bash

echo "🚀 开始安装 Ollama..."

# 检查是否已安装
if command -v ollama &> /dev/null; then
    echo "✅ Ollama 已安装"
else
    echo "⬇️ 正在下载并安装 Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# 启动 Ollama 服务
echo "🔧 启动 Ollama 服务..."
ollama serve &
sleep 3

# 下载推荐模型 (根据 32GB 内存，推荐 7B-14B 大小的模型)
echo "⬇️ 下载推荐模型 (Qwen2.5 7B - 中文支持好，速度快)..."
ollama pull qwen2.5:7b

echo ""
echo "✅ 安装完成！"
echo ""
echo "使用方法："
echo "  1. 终端输入：ollama run qwen2.5:7b"
echo "  2. 或者用其他客户端连接 http://localhost:11434"
echo ""
echo "其他可选模型："
echo "  - ollama pull llama3.2:3b  (更小更快)"
echo "  - ollama pull qwen2.5:14b  (更聪明，但稍慢)"
echo "  - ollama pull gemma2:9b   (Google 开源模型)"
echo ""
echo "💡 提示：模型下载可能需要一些时间，取决于网络速度"
