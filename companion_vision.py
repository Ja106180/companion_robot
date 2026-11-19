"""
全息伴生舱 · 全息视觉之眼引擎 (Companion Vision Engine)
========================================================
核心特性（0 本地显卡显存占用 · 极速响应）：
1. 电脑自带摄像头实景抓拍 (OpenCV `/dev/video0`)：
   - 自动预热对焦与自动曝光补偿，抓拍高清画面后立即释放摄像头（保护隐私、不常驻）。
2. 当前电脑页面 / 屏幕极速截图 (Python MSS `DISPLAY=:1/:0`)：
   - 5~20ms 截取当前电脑实时画面并智能缩放压缩为高清 JPEG Base64，电脑与手机秒速同步显示。
3. AI 视觉识图点评（支持云端免费多模态大模型 VLM + OpenCV 零显存实时人脸/光线/窗口感知双引擎）：
   - 若配置了支持视觉的云端接口（如硅基流动免费 Qwen2.5-VL、智谱免费 GLM-4V-Flash、通义千问 VL），自动调用多模态识图。
   - 若当前仅使用纯文本模型（如 DeepSeek-V3），自动通过 OpenCV 识别人脸位置、室内光线、主色调及当前活动窗口标题，生成真实准确的视觉感知报告供大脑点评。
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import subprocess
import time
import urllib.request
from typing import Any

import cv2
import mss
import numpy as np

logger = logging.getLogger("CompanionVision")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [Vision] %(message)s"))
    logger.addHandler(_h)


class CompanionVisionEngine:
    """全息视觉抓拍、屏幕截图与智能识图引擎"""

    def __init__(self) -> None:
        self._face_cascade = None
        try:
            xml_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            if os.path.exists(xml_path):
                self._face_cascade = cv2.CascadeClassifier(xml_path)
        except Exception as exc:
            logger.warning(f"Face cascade init warning: {exc}")

    # ------------------------------------------------------------------
    # 1. 意图识别：判断用户是否在要求“拍摄像头照片”或“发当前页面/屏幕截图”
    # ------------------------------------------------------------------
    def detect_vision_intent(self, user_text: str) -> str | None:
        """
        返回:
          - 'camera': 用户要求调用电脑摄像头拍照/看看用户/人脸识别
          - 'switch_face': 用户要求切换看旁边的人/换个人看
          - 'screen': 用户要求截取当前电脑页面/屏幕画面发过来
          - None: 普通对话
        """
        t = user_text.strip()
        if not t:
            return None

        # 显式切换人脸焦点意图
        switch_face_pattern = re.compile(
            r"(看.*旁边.*谁|看.*旁边.*人|看看旁边|看看另一个人|看.*另外.*人|换个人看|看下一个人|切换目标|旁边是谁)"
        )
        if switch_face_pattern.search(t):
            return "switch_face"

        # 显式屏幕/页面截图意图
        screen_pattern = re.compile(
            r"(当前页面|当前画面|电脑屏幕|桌面截图|屏幕截图|页面图片|页面截图|画面截图|截个图|截张图|截屏|发.*页面.*图|看.*电脑画面|看.*当前页面)"
        )
        if screen_pattern.search(t):
            return "screen"

        # 显式摄像头/拍照/看人意图与人脸识别
        camera_pattern = re.compile(
            r"(摄像头|拍张照|拍一张|拍照|拍个照|自拍|拍.*画面|看.*我在干嘛|看.*我穿什么|看.*我手里|看.*电脑前|发.*照片|看看我是谁|认得我吗|我是谁|认出我|看.*前面.*谁|看看镜头|人脸识别|看看我)"
        )
        if camera_pattern.search(t):
            return "camera"

        # 泛指“发张图片过来 / 发个图看看” -> 默认发当前页面截图（最贴合“发一张当前页面的图片过来”）
        generic_img_pattern = re.compile(r"(发.*一张.*图|发.*图片.*过来|发张图看看|发个图片)")
        if generic_img_pattern.search(t):
            return "screen"

        return None

    # ------------------------------------------------------------------
    # 2. 电脑自带摄像头高清抓拍 (OpenCV)
    # ------------------------------------------------------------------
    def capture_webcam(self, target_index: int | None = None) -> dict[str, Any]:
        t0 = time.time()
        cap = None
        try:
            # 优先从常驻的 companion_sentinel 内存中直接获取最新帧（0ms极速抓拍且无硬件占用冲突）
            frame = None
            try:
                from companion_sentinel import companion_sentinel
                frame = companion_sentinel.get_latest_clean_frame()
            except Exception:
                frame = None

            if frame is None:
                # 后备方案：直接开启 OpenCV 抓拍
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(1)
                if not cap.isOpened():
                    return {
                        "success": False,
                        "error": "未检测到可用的电脑摄像头（/dev/video0 未就绪或被其他程序占用）",
                    }

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

                for _ in range(6):
                    ret, f = cap.read()
                    if ret and f is not None:
                        frame = f
                    time.sleep(0.02)
                cap.release()
                cap = None

            if frame is None:
                return {"success": False, "error": "摄像头读取画面帧失败"}

            h, w = frame.shape[:2]

            # 轻量级 OpenCV 基础感知（光线亮度分析）
            scene_desc = self._analyze_camera_frame_opencv(frame)

            # InsightFace 深度人脸识别与多目标视线锁定
            face_analysis = None
            try:
                from companion_face import companion_face
                face_analysis = companion_face.analyze_frame(frame, target_index=target_index)
            except Exception as face_err:
                logger.warning(f"Face analysis error: {face_err}")

            chip_title = f"全息视线对焦 · 电脑摄像头实景 ({w}×{h})"
            if face_analysis and face_analysis.get("locked_target"):
                lt = face_analysis["locked_target"]
                chip_title = f"视线锁定 · {lt['display_badge']}"
                scene_desc += f" 【人脸锁定】：{lt['display_badge']}。"
                if face_analysis.get("faces_count", 0) > 1:
                    scene_desc += f"（画面中共有 {face_analysis['faces_count']} 位人物，已为您选择性对焦目标 {lt['target_index']+1}）"

            # 编码为高质量 JPEG Base64
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 86]
            ok, buf = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                return {"success": False, "error": "摄像头图像编码失败"}

            jpg_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            ms = int((time.time() - t0) * 1000)

            return {
                "success": True,
                "mode": "camera",
                "width": w,
                "height": h,
                "elapsed_ms": ms,
                "image_data_uri": f"data:image/jpeg;base64,{jpg_b64}",
                "raw_b64": jpg_b64,
                "opencv_summary": scene_desc,
                "chip_title": chip_title,
                "face_analysis": face_analysis,
            }
        except Exception as exc:
            logger.warning(f"Webcam capture error: {exc}")
            return {"success": False, "error": str(exc)}
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    def _analyze_camera_frame_opencv(self, frame: np.ndarray) -> str:
        """零显存快速分析摄像头画面：亮度、人物是否在电脑前、画面色调"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_bright = float(np.mean(gray))
            if avg_bright < 45:
                light_str = "室内光线较暗（夜间或关灯环境）"
            elif avg_bright < 110:
                light_str = "室内光线柔和适中"
            elif avg_bright < 185:
                light_str = "室内光线明亮充足"
            else:
                light_str = "画面光线非常明亮"

            face_str = "镜头前暂未检测到正脸（主人可能在侧面、离开座位或手持物品）"
            if self._face_cascade is not None:
                faces = self._face_cascade.detectMultiScale(
                    gray, scaleFactor=1.15, minNeighbors=5, minSize=(48, 48)
                )
                if len(faces) == 1:
                    x, y, fw, fh = faces[0]
                    h, w = frame.shape[:2]
                    cx = (x + fw / 2) / w
                    pos = "画面正中央" if 0.32 <= cx <= 0.68 else ("画面偏左侧" if cx < 0.32 else "画面偏右侧")
                    dist_desc = "距离镜头较近、清晰可见" if (fw / w) > 0.22 else "端坐在电脑前"
                    face_str = f"检测到主人正坐在电脑镜头前（位于{pos}，{dist_desc}）"
                elif len(faces) > 1:
                    face_str = f"检测到电脑镜头前有 {len(faces)} 个人同框"

            return f"摄像头实景感知：{face_str}；环境状态：{light_str}。"
        except Exception:
            return "摄像头已成功抓拍电脑前实时画面。"

    # ------------------------------------------------------------------
    # 3. 当前电脑页面 / 屏幕极速截图 (Python MSS)
    # ------------------------------------------------------------------
    def capture_screen(self) -> dict[str, Any]:
        t0 = time.time()
        try:
            if "DISPLAY" not in os.environ:
                os.environ["DISPLAY"] = ":1"

            with mss.MSS() as sct:
                mon = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                shot = sct.grab(mon)
                orig_w, orig_h = shot.size

                # BGRA -> BGR
                img_np = np.array(shot)
                frame_bgr = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)

                # 智能缩放到宽度 1440 以内，保证高清字迹同时让手机端秒速加载
                target_w = 1440
                if orig_w > target_w:
                    scale = target_w / float(orig_w)
                    target_h = int(orig_h * scale)
                    frame_bgr = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)

                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 84]
                ok, buf = cv2.imencode(".jpg", frame_bgr, encode_params)
                if not ok:
                    return {"success": False, "error": "屏幕截图编码失败"}

                jpg_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                ms = int((time.time() - t0) * 1000)

                active_win = self._get_active_window_title()
                win_desc = f"当前电脑前台窗口为「{active_win}」" if active_win else "当前展示着全息伴生舱与电脑桌面实时画面"

                return {
                    "success": True,
                    "mode": "screen",
                    "width": orig_w,
                    "height": orig_h,
                    "elapsed_ms": ms,
                    "image_data_uri": f"data:image/jpeg;base64,{jpg_b64}",
                    "raw_b64": jpg_b64,
                    "opencv_summary": f"电脑屏幕实时截图已完成（原始分辨率 {orig_w}×{orig_h}），{win_desc}。",
                    "chip_title": f"全息视觉已截取 · 当前电脑页面实时画面 ({orig_w}×{orig_h} · {ms}ms)",
                }
        except Exception as exc:
            logger.warning(f"Screen capture error: {exc}")
            return {"success": False, "error": str(exc)}

    def _get_active_window_title(self) -> str:
        try:
            env = os.environ.copy()
            env.setdefault("DISPLAY", ":1")
            out = subprocess.check_output(
                ["xprop", "-root", "_NET_ACTIVE_WINDOW"],
                env=env,
                text=True,
                timeout=1.0,
                stderr=subprocess.DEVNULL,
            )
            m = re.search(r"0x[0-9a-fA-F]+", out)
            if m and m.group(0) != "0x0":
                win_id = m.group(0)
                w_out = subprocess.check_output(
                    ["xprop", "-id", win_id, "_NET_WM_NAME"],
                    env=env,
                    text=True,
                    timeout=1.0,
                    stderr=subprocess.DEVNULL,
                )
                m_title = re.search(r'"([^"]+)"', w_out)
                if m_title:
                    return m_title.group(1)
        except Exception:
            pass
        return ""

    # ------------------------------------------------------------------
    # 4. 可选云端多模态视觉大模型识图 (0 本地显存占用)
    # ------------------------------------------------------------------
    def describe_with_cloud_vlm(
        self,
        raw_b64: str,
        user_text: str,
        mode: str,
        llm_cfg: dict[str, Any],
    ) -> str | None:
        """
        若用户配置了支持多模态视觉的云端密钥（如硅基流动、智谱、阿里云百炼或当前模型本身支持 VL），
        自动调用云端视觉模型对抓拍图片进行精准描述（耗时约 1~2 秒，本机 0 显存占用）。
        """
        saved_keys = llm_cfg.get("saved_keys") or {}
        cur_provider = llm_cfg.get("provider") or ""
        cur_key = (llm_cfg.get("api_key") or "").strip()

        # 选择可用的云端多模态视觉通道
        vlm_endpoint = ""
        vlm_key = ""
        vlm_model = ""

        # 1. 若有智谱 Key -> 优先用官方永久免费的 glm-4v-flash
        if cur_provider == "zhipu" and cur_key:
            vlm_endpoint = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
            vlm_key = cur_key
            vlm_model = "glm-4v-flash"
        elif saved_keys.get("zhipu"):
            vlm_endpoint = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
            vlm_key = saved_keys["zhipu"]
            vlm_model = "glm-4v-flash"
        # 2. 若有硅基流动 Key -> 使用 Qwen/Qwen2.5-VL-7B-Instruct
        elif cur_provider == "siliconflow" and cur_key:
            vlm_endpoint = "https://api.siliconflow.cn/v1/chat/completions"
            vlm_key = cur_key
            vlm_model = "Qwen/Qwen2.5-VL-7B-Instruct"
        elif saved_keys.get("siliconflow"):
            vlm_endpoint = "https://api.siliconflow.cn/v1/chat/completions"
            vlm_key = saved_keys["siliconflow"]
            vlm_model = "Qwen/Qwen2.5-VL-7B-Instruct"
        # 3. 若有阿里云百炼 Key -> 使用 qwen-vl-plus
        elif cur_provider == "aliyun" and cur_key:
            vlm_endpoint = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
            vlm_key = cur_key
            vlm_model = "qwen-vl-plus"
        elif saved_keys.get("aliyun"):
            vlm_endpoint = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
            vlm_key = saved_keys["aliyun"]
            vlm_model = "qwen-vl-plus"
        # 4. 若当前自定义模型名称包含 vl / vision / gpt-4o / gemini
        elif cur_key and any(k in (llm_cfg.get("cloud_model") or "").lower() for k in ("vl", "vision", "gpt-4o", "gemini")):
            base_url = (llm_cfg.get("base_url") or "").rstrip("/")
            vlm_endpoint = f"{base_url}/chat/completions"
            vlm_key = cur_key
            vlm_model = llm_cfg.get("cloud_model") or ""

        if not (vlm_endpoint and vlm_key and vlm_model):
            return None

        try:
            prompt = (
                f"用简短的中文（1到2句话）客观描述这张{'电脑摄像头实拍照片' if mode == 'camera' else '电脑屏幕截图'}里的核心内容，"
                f"重点关注用户的问题：「{user_text}」。"
            )
            payload = {
                "model": vlm_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{raw_b64}"}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                "max_tokens": 150,
                "temperature": 0.4,
            }
            req = urllib.request.Request(
                vlm_endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {vlm_key}",
                    "User-Agent": "BreezeCompanion/1.0",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                choices = data.get("choices") or []
                if choices:
                    desc = (choices[0].get("message") or {}).get("content") or ""
                    return desc.strip()
        except Exception as exc:
            logger.warning(f"Cloud VLM description skipped/failed: {exc}")
        return None

    # ------------------------------------------------------------------
    # 5. 统一入口：根据用户发言自动执行抓拍/截图并生成视觉上下文
    # ------------------------------------------------------------------
    def analyze_and_capture(
        self,
        user_text: str,
        llm_cfg: dict[str, Any],
        enable_ai_vision: bool = True,
    ) -> dict[str, Any] | None:
        intent = self.detect_vision_intent(user_text)
        if not intent:
            return None

        logger.info(f"[Vision] Triggered vision capture intent: {intent}")
        if intent == "switch_face":
            from companion_face import companion_face
            switch_res = companion_face.switch_to_next_target()
            next_idx = switch_res.get("locked_index")
            res = self.capture_webcam(target_index=next_idx)
            intent = "camera"
        elif intent == "camera":
            res = self.capture_webcam()
        else:
            res = self.capture_screen()

        if not res.get("success"):
            return {
                "success": False,
                "mode": intent,
                "chip_title": f"全息视觉提示 · {res.get('error', '抓拍失败')}",
                "context_text": f"【全息视觉状态】：尝试执行{'摄像头抓拍' if intent == 'camera' else '页面截图'}时遇到问题：{res.get('error')}。请温柔地告知主人。",
            }

        visual_detail = res.get("opencv_summary", "")
        if enable_ai_vision and res.get("raw_b64"):
            vlm_desc = self.describe_with_cloud_vlm(
                res["raw_b64"], user_text, intent, llm_cfg
            )
            if vlm_desc:
                visual_detail += f" AI视觉深度识图结果：{vlm_desc}"

        face_greeting_hint = ""
        fa = res.get("face_analysis")
        if fa and fa.get("locked_target"):
            lt = fa["locked_target"]
            greeting = fa.get("greeting_text", "")
            face_greeting_hint = (
                f"\n【人脸视线锁定指令】：系统已锁定当前目标【{lt['label']}】（{lt['display_badge']}）。"
                f"请在回答中必须包含问候语义「{greeting}」，并以你的萌宠性格亲切自然地与对方交流！"
            )

        action_name = "电脑自带摄像头实景照片" if intent == "camera" else "当前电脑页面/屏幕高清截图"
        res["context_text"] = (
            f"【全息视觉已成功执行】：系统刚刚已经帮主人抓拍了一张「{action_name}」，"
            f"并且这张图片已经实时显示在主人的电脑主屏和手机聊天窗口里了！\n"
            f"【画面实时感知信息】：{visual_detail}"
            f"{face_greeting_hint}"
        )
        return res


companion_vision = CompanionVisionEngine()
