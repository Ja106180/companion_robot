# 全息伴生萌宠 · 离线全链路语音与多模态智能体系统
# Holographic Companion Robot

> **基于 Breeze-TTS-2 + SenseVoice-Small + 3D-Speaker 声纹门禁 + Mem0 长期记忆 + 全息毛绒伙伴的桌面全能伴生智能体**

本项目是一个完整的端到端桌面伴生灵宠交互系统，集成了**离线超快语音合成 (TTS)**、**四层抗噪离线语音识别 (ASR)**、**主人专属声纹门禁 (Speaker Verification)**、**长期记忆沉淀中枢 (Mem0)**、**手机无线全息遥控**、**实时视觉感知**与**无密钥全网检索**能力。

---

## 核心特性一览

### 1. 6 位毛绒伙伴与专属性格声线
- **棉棉 (Luma)**：温柔治愈系大耳白兔，贴心棉花糖般陪伴，语速轻缓柔和。
- **桃桃 (Pipi)**：活泼软萌粉粉兔，好奇心旺盛、元气满满、音调轻快甜美。
- **芋圆 (Taigo)**：傲娇小猫咪，外冷内热，偶尔吐槽，音色清脆通透。
- **极光 (Auri)**：理性博学小狐狸，条理清晰、知识渊博，音色沉稳清朗。
- **暖暖 (Nuannuan)**：呆萌黏人小考拉，软糯依恋、爱撒娇，音色稚嫩奶糯。
- **多多 (Duoduo)**：机灵幽默小柯基，风趣乐观、忠实守护，音色明朗少年感。

### 2. 四层抗噪离线语音识别 (Companion ASR)
- **第 1 层 · GTCRN 神经网络降噪**：对麦克风输入环境噪音、键盘敲击声进行瞬时擦除。
- **第 2 层 · Silero VAD 真人声检测**：毫秒级切分有效语音，拦截呼吸声与无意义短促杂音（干掉 `"Yeah."` / `"H."` 等幻觉词）。
- **第 3 层 · 3D-Speaker CAM++ 主人声纹门禁**：仅在主人开口时采纳识别，旁人说话及萌宠自身发音全部直接拒之门外（可在界面一键录入/切换）。
- **第 4 层 · SenseVoice-Small (INT8) 离线转写 + 伙伴同音字点名纠偏**：纯 CPU 极速转写（~60ms 延迟，零显存占用），伙伴名字呼叫（“棉棉/绵绵/桃桃”）自动精准对齐。
- **双保险防自循环回声消除**：精准 AudioBuffer 播放计数 + 1.5s 智能冷却拦截，彻底杜绝“电脑自己听自己说话”。

### 3. Breeze-TTS-2 Vulkan 离线超低延迟语音合成
- 基于 Vulkan GPU 加速引擎与 Q4_K 量化模型，首包音频生成延迟低至 **50~150ms**。
- 支持 6 种声线音色特征实时变换，伴随全身毛绒呼吸动效、眨眼、摆动与全息状态台。

### 4. Mem0 长期记忆银行 (二阶段槽位去重架构)
- 本地嵌入式 Qdrant 向量数据库 + FastEmbed (BGE-small-zh)，**零显存占用**。
- **二阶段事实提炼与冲突消解**：仅从最新输入中提炼原子事实，同名槽位原地更新（UPDATE），杜绝幻觉复读。

### 5. 手机无线遥控与全息中枢 (PC & Mobile 实时同步)
- **通道一（局域网 Wi-Fi 直连）**：手机与电脑同 Wi-Fi 下扫码秒连，零公网依赖，永不掉线。
- **通道二（Cloudflare 免费公网穿透）**：支持手机 4G/5G 远程随身控，内置 Watchdog 自动健康巡检保活。

### 6. 视觉之眼与免密全网感知
- 电脑摄像头实时抓拍（OpenCV 对焦曝光自适应）+ 屏幕实时极速截屏（MSS 5~15ms）。
- 实时全国城市天气（Open-Meteo + wttr.in 双擎）+ 每日全网热搜（DailyHot 聚合检索）。

---

## 硬件与系统要求

- **操作系统**：Linux (Ubuntu 20.04 / 22.04 / 24.04, Debian 11+, Linux Mint 等推荐)
- **显卡要求**：推荐 NVIDIA 独立显卡（如 RTX 4060 / 3060 等，支持 Vulkan / CUDA）；显存占用仅需约 2~3GB。
- **CPU / 内存**：8 核 CPU / 16GB 内存及以上推荐。
- **Python 版本**：Python 3.10 或更高版本。

---

## 新电脑从 GitHub 克隆后快速部署

在任何一台新电脑上，只需通过以下几步即可完全一致地运行本项目：

### 1. 克隆代码仓库
```bash
git clone git@github.com:Ja106180/companion_robot.git
cd companion_robot
```

### 2. 方式一：一键自动化安装向导（强烈推荐）
项目已编写全自动部署脚本，会自动安装依赖、创建虚拟环境并多线程高速下载超大模型：
```bash
bash setup.sh
```
部署完成后，直接启动：
```bash
./start.sh
```

---

### 3. 方式二：手动分步执行安装

如果你希望手动逐步掌控每一步，请参考以下完整命令：

#### 步骤 1：安装 Linux 系统底层音频与多媒体依赖
```bash
sudo apt update
sudo apt install -y ffmpeg libvulkan1 pulseaudio-utils alsa-utils libsndfile1 portaudio19-dev
```

#### 步骤 2：创建并激活 Python 虚拟环境
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 步骤 3：安装 Python 依赖库
```bash
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 步骤 4：下载超大模型权重
由于 GitHub 单文件有 100MB 上限，Breeze-TTS-2 (2.53GB) 与 SenseVoice (239MB) 未直接存放在 Git 仓库中。执行以下命令即可从 ModelScope 魔搭社区多线程秒级高速补全：
```bash
python3 download_models.py
```
> 下载器默认使用 8 线程并发，国内网络通常在 1~3 分钟内即可下载完成。

#### 步骤 5：安装并配置本地大模型（Ollama）
本项目默认使用本地轻量高效的 `qwen2.5:3b` 作为大脑核心（也可在界面中随时切换为云端 DeepSeek / 通义千问等）：
```bash
# 1. 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. 启动 Ollama 服务并拉取 3B 模型
ollama serve &
ollama pull qwen2.5:3b
```

#### 步骤 6：配置执行权限并启动项目
```bash
chmod +x bin/cloudflared bin/breeze-tts-2-v0.1.7-linux-x64-vulkan/* start.sh
python3 app.py
```

---

## 启动与使用指南

1. **进入系统主控台**：
   - 打开电脑浏览器访问：`http://127.0.0.1:7860`
2. **连接手机遥控端**：
   - 点击右上角【手机扫码遥控】按钮打开无线中枢面板。
   - **同一 Wi-Fi**：直接用手机相机或浏览器扫描“通道一（局域网）二维码”，即刻开启手机遥控。
   - **4G/5G 流量**：扫描“通道二（公网穿透）二维码”，随时随地跨网络无线连接。
3. **录入主人专属声纹（声纹门禁）**：
   - 在主界面点击【声纹录入】开关。
   - 面对麦克风自然说出 3~5 秒任意一句话（如“你好棉棉，我是你的主人”）。
   - 提取成功后，声纹门禁正式生效，旁人和外界电视杂音均被自动屏蔽！

---

## 项目工程结构目录

```text
companion_robot/
├── app.py                     # 系统核心主程序 (FastAPI 服务、多轮对话调度与状态机)
├── companion_asr.py           # 四层离线语音识别流水线 (GTCRN + Silero VAD + CAM++ + SenseVoice)
├── companion_memory.py        # Mem0 长期记忆中枢 (二阶段槽位去重消解、Qdrant 向量检索)
├── companion_mobile.py        # 手机无线中枢服务 (SSE 全双工同步、Cloudflare 穿透与看门狗)
├── companion_vision.py        # 视觉之眼引擎 (OpenCV 摄像头抓拍、MSS 极速截屏、视觉感知)
├── companion_web.py           # 实时全网无秘感知引擎 (全国气象、DailyHot 热榜、搜索聚合)
├── download_models.py         # 超大模型权重多线程高速下载器
├── setup.sh                   # 新电脑一键部署脚本
├── start.sh                   # 一键启动脚本
├── requirements.txt           # Python 核心依赖清单
├── bin/                       # 编译可执行程序目录
│   ├── breeze-tts-2-.../      # Breeze-TTS-2 Vulkan Linux x64 语音合成推理引擎
│   └── cloudflared            # Cloudflare 穿透工具二进制
├── data/                      # 持久化数据与模型组件
│   ├── asr_guards/            # ASR 三重神经护栏权重 (GTCRN, Silero VAD, CAM++ 声纹模型)
│   ├── sensevoice_int8/       # SenseVoice 离线词表 tokens.txt
│   ├── llm_config.json        # 大模型配置文件 (支持本地 Ollama 与云端 DeepSeek)
│   └── mem0_history.db        # 伙伴长期记忆存储数据库
├── models/                    # 语音合成模型目录
│   └── ref_voice.wav          # 伴生萌宠基准音色参考音频
├── static/                    # 前端毛绒动态素材、SVG 表情与静态资源
└── templates/                 # 网页模板
    ├── index.html             # 电脑端全息桌面主屏
    └── mobile.html            # 手机端无线手持全功能交互界面
```

---

## 常见问题与排查 (FAQ)

### Q1：启动时提示缺少某个模型权重文件？
- 请运行 `python3 download_models.py`，脚本会自动检测缺失项并以极速多线程下载到对应目录。

### Q2：启动提示麦克风打开失败？
- 请确认 Linux 是否安装了音频工具：`sudo apt install -y pulseaudio-utils alsa-utils`。
- 如果在虚拟机或 WSL2 中运行，请确保宿主机的麦克风设备已映射到 Linux 环境。

### Q3：手机扫码提示“网络出错，无法显示该页面”？
- **优先推荐**：手机和电脑连接同一个 Wi-Fi，扫描**通道一（局域网）二维码**，永不超时，连接最稳。
- 若使用**通道二（公网穿透）**：系统内置看门狗会自动检测并刷新；如遇网络切换，可在电脑面板上点击【重新生成穿透】获取最新二维码。

---

## 开源协议与鸣谢

本项目基于以下优秀的开源项目构建与深度融合：
- [Breeze-TTS-2](https://github.com/breezeblue-ai/breeze-tts) / [Breeze-TTS-2.cpp](https://github.com/HoppouAI/Breeze-TTS-2.cpp)
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) & [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- [3D-Speaker](https://github.com/alibaba-damo-academy/3D-Speaker) (阿里达摩院 CAM++ 声纹识别)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [Mem0](https://github.com/mem0ai/mem0)
- [FastAPI](https://github.com/tiangolo/fastapi)
