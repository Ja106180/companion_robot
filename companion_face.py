"""
companion_face.py - 全息伴生舱 · InsightFace 离线人脸识别与视线对焦中枢
========================================================================
核心特性（0 显卡显存占用 · 纯 ONNX 极速推理 · 毫秒级锁定）：
1. 工业级 ArcFace 512 维特征向量比对（阈值 0.60 判定为同一人）。
2. 单人/多人多策略精准聚焦：
   - 策略一（主人优先锁定）：无论画面中有多少人，只要匹配到主人（相似度 >= 0.6），自动锁定主人。
   - 策略二（近场 C 位最大面积锁定）：若全为陌生人，自动锁定离镜头最近、面积最大的人脸。
   - 策略三（高科技目镜单点切换）：画面有多人时，支持在前端点选【目标 1】【目标 2】，或语音“看看旁边那个人是谁”自动切焦。
3. 问候与情感反馈：
   - 识别为主人：“主人，你回来啦！”
   - 识别为陌生人：“检测到新朋友在镜头前”
   - 识别为熟悉好友：“检测到{姓名}在镜头前，你好呀！”
4. 人脸面孔中枢管理：
   - 摄像头一键抓拍录入主人
   - 上传照片录入熟悉好友
   - 本地持久化保存 (data/face_vault.json)，100% 离线隐私无上传
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("CompanionFace")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [Face] %(message)s"))
    logger.addHandler(_h)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

VAULT_PATH = DATA_DIR / "face_vault.json"
INSIGHTFACE_ROOT = DATA_DIR / "insightface"
SIMILARITY_THRESHOLD = 0.60


class CompanionFaceEngine:
    """基于 InsightFace (SCRFD + ArcFace) 的离线人脸比对与视线对焦中枢"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._app = None
        self._vault: dict[str, Any] = {"master": None, "friends": []}
        self._last_detected_faces: list[dict[str, Any]] = []
        self._last_locked_index: int = 0
        self._load_vault()
        self._init_model_async()

    def _init_model_async(self) -> None:
        """异步在后台加载模型，避免阻塞服务启动"""
        def _worker() -> None:
            self._ensure_model_loaded()

        t = threading.Thread(target=_worker, daemon=True, name="FaceModelLoader")
        t.start()

    def _ensure_model_loaded(self) -> bool:
        if self._app is not None:
            return True
        with self._lock:
            if self._app is not None:
                return True
            try:
                from insightface.app import FaceAnalysis

                t0 = time.time()
                # 仅加载 detection(SCRFD) 和 recognition(MobileFaceNet/ArcFace)，速度最快且内存最小
                app = FaceAnalysis(
                    name="buffalo_s",
                    root=str(INSIGHTFACE_ROOT),
                    allowed_modules=["detection", "recognition"],
                )
                app.prepare(ctx_id=-1, det_size=(640, 640))
                self._app = app
                ms = int((time.time() - t0) * 1000)
                logger.info(f"InsightFace engine loaded successfully in {ms}ms")
                return True
            except Exception as exc:
                logger.error(f"Failed to load InsightFace model: {exc}")
                return False

    # ==========================================================
    # 本地面孔数据库 (Face Vault)
    # ==========================================================
    def _load_vault(self) -> None:
        if not VAULT_PATH.is_file():
            self._vault = {"master": None, "friends": []}
            return
        try:
            with open(VAULT_PATH, "r", encoding="utf-8") as f:
                self._vault = json.load(f)
            master_name = self._vault.get("master", {}).get("name") if self._vault.get("master") else None
            friends_count = len(self._vault.get("friends", []))
            logger.info(f"Loaded Face Vault: master={master_name}, friends={friends_count}")
        except Exception as exc:
            logger.warning(f"Error loading face vault: {exc}")
            self._vault = {"master": None, "friends": []}

    def _save_vault(self) -> None:
        try:
            with open(VAULT_PATH, "w", encoding="utf-8") as f:
                json.dump(self._vault, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.error(f"Error saving face vault: {exc}")

    def get_vault_info(self) -> dict[str, Any]:
        """获取已录入面孔信息清单"""
        with self._lock:
            master = self._vault.get("master")
            friends = self._vault.get("friends", [])
            return {
                "success": True,
                "has_master": master is not None,
                "master": {
                    "id": master.get("id"),
                    "name": master.get("name", "主人"),
                    "created_at": master.get("created_at"),
                    "thumbnail": master.get("thumbnail"),
                }
                if master
                else None,
                "friends": [
                    {
                        "id": f.get("id"),
                        "name": f.get("name"),
                        "created_at": f.get("created_at"),
                        "thumbnail": f.get("thumbnail"),
                    }
                    for f in friends
                ],
                "threshold": SIMILARITY_THRESHOLD,
            }

    def delete_face(self, face_id: str) -> dict[str, Any]:
        """删除特定人脸（主人或好友）"""
        with self._lock:
            master = self._vault.get("master")
            if master and master.get("id") == face_id:
                self._vault["master"] = None
                self._save_vault()
                logger.info(f"Deleted master face: {face_id}")
                return {"success": True, "message": "已清除主人面孔"}

            friends = self._vault.get("friends", [])
            new_friends = [f for f in friends if f.get("id") != face_id]
            if len(new_friends) != len(friends):
                self._vault["friends"] = new_friends
                self._save_vault()
                logger.info(f"Deleted friend face: {face_id}")
                return {"success": True, "message": "已删除该面孔记录"}

            return {"success": False, "error": "未找到指定的人脸记录"}

    # ==========================================================
    # 人脸录入（拍照录入 / 照片上传录入）
    # ==========================================================
    def enroll_from_frame(
        self,
        frame: np.ndarray,
        name: str = "主人",
        is_master: bool = True,
    ) -> dict[str, Any]:
        """从 OpenCV BGR 图像帧录入人脸"""
        if not self._ensure_model_loaded():
            return {"success": False, "error": "人脸分析引擎正在加载，请稍候"}

        h, w = frame.shape[:2]
        faces = self._app.get(frame)
        if not faces:
            return {"success": False, "error": "照片中未检测到清晰人脸，请保持面部正对镜头、光线充足"}

        # 若照片有多人，取面积最大的正脸
        faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        face = faces[0]
        embedding = face.embedding
        if embedding is None:
            return {"success": False, "error": "人脸特征提取失败"}

        # 归一化特征向量
        emb_norm = embedding / np.linalg.norm(embedding)

        # 裁剪生成小型缩略图 Base64 (120x120)
        thumb_b64 = ""
        try:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            # 适当外扩边缘
            pad_x = int((x2 - x1) * 0.2)
            pad_y = int((y2 - y1) * 0.2)
            nx1 = max(0, x1 - pad_x)
            ny1 = max(0, y1 - pad_y)
            nx2 = min(w, x2 + pad_x)
            ny2 = min(h, y2 + pad_y)
            crop = frame[ny1:ny2, nx1:nx2]
            if crop.size > 0:
                crop_resized = cv2.resize(crop, (120, 120), interpolation=cv2.INTER_AREA)
                _, buf = cv2.imencode(".jpg", crop_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                thumb_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
        except Exception as exc:
            logger.debug(f"Thumbnail error: {exc}")

        face_id = f"face_{int(time.time())}_{name}"
        record = {
            "id": face_id,
            "name": name.strip() or ("主人" if is_master else "熟悉朋友"),
            "is_master": is_master,
            "created_at": time.strftime("%Y-%m-%d %H:%M"),
            "embedding": emb_norm.tolist(),
            "thumbnail": thumb_b64,
        }

        with self._lock:
            if is_master:
                self._vault["master"] = record
            else:
                self._vault.setdefault("friends", []).append(record)
            self._save_vault()

        role_str = "主人面孔" if is_master else f"朋友【{name}】面孔"
        logger.info(f"Enrolled {role_str} successfully (id={face_id})")
        return {
            "success": True,
            "message": f"已成功录入{role_str}",
            "face_id": face_id,
            "name": record["name"],
            "thumbnail": thumb_b64,
        }

    def enroll_from_image_bytes(
        self,
        image_bytes: bytes,
        name: str = "主人",
        is_master: bool = True,
    ) -> dict[str, Any]:
        """从任意图片二进制数据录入"""
        try:
            arr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return {"success": False, "error": "图片解码失败，请上传有效的 JPG/PNG 格式图片"}
            return self.enroll_from_frame(frame, name=name, is_master=is_master)
        except Exception as exc:
            logger.warning(f"Enroll image error: {exc}")
            return {"success": False, "error": f"录入失败: {exc}"}

    # ==========================================================
    # 人脸检测、多目标比对与焦点选择
    # ==========================================================
    def analyze_frame(
        self,
        frame: np.ndarray,
        target_index: int | None = None,
    ) -> dict[str, Any]:
        """
        核心人脸分析：
        1. 识别画面中所有面孔
        2. 与数据库比对（主人 >= 0.60 优先；否则近场 C 位最大面积锁定；或指定 target_index 锁定）
        3. 返回高科技视线对焦框参数与纯净简洁状态（主人 · 置信度 0.82 / 新朋友 / 熟悉好友）
        """
        if not self._ensure_model_loaded():
            return {
                "success": False,
                "faces_count": 0,
                "error": "人脸分析引擎正在初始化，请稍候",
            }

        h, w = frame.shape[:2]
        t0 = time.time()
        detected = self._app.get(frame)
        latency_ms = int((time.time() - t0) * 1000)

        if not detected:
            with self._lock:
                self._last_detected_faces = []
                self._last_locked_index = 0
            return {
                "success": True,
                "faces_count": 0,
                "latency_ms": latency_ms,
                "locked_target": None,
                "targets": [],
                "greeting_text": "镜头前未检测到人脸",
            }

        with self._lock:
            master_rec = self._vault.get("master")
            friends_recs = self._vault.get("friends", [])

        # 解析所有人脸的特征与相似度
        parsed_targets: list[dict[str, Any]] = []
        for i, face in enumerate(detected):
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            area = max(0, (x2 - x1) * (y2 - y1))

            emb = face.embedding
            master_score = 0.0
            best_friend_name = ""
            best_friend_score = 0.0

            if emb is not None:
                emb_norm = emb / np.linalg.norm(emb)
                # 与主人比对
                if master_rec and master_rec.get("embedding"):
                    m_vec = np.array(master_rec["embedding"], dtype=np.float32)
                    master_score = float(np.dot(emb_norm, m_vec))

                # 与熟悉好友比对
                for f in friends_recs:
                    if f.get("embedding"):
                        f_vec = np.array(f["embedding"], dtype=np.float32)
                        s = float(np.dot(emb_norm, f_vec))
                        if s > best_friend_score:
                            best_friend_score = s
                            best_friend_name = f.get("name", "好友")

            # 判定身份与置信度
            is_master = master_score >= SIMILARITY_THRESHOLD
            is_friend = (not is_master) and (best_friend_score >= SIMILARITY_THRESHOLD)

            if is_master:
                identity = "master"
                label = "主人"
                display_badge = f"主人 · 置信度 {master_score:.2f}"
                confidence = master_score
            elif is_friend:
                identity = "friend"
                label = best_friend_name
                display_badge = f"{best_friend_name} · 置信度 {best_friend_score:.2f}"
                confidence = best_friend_score
            else:
                identity = "stranger"
                label = "新朋友"
                display_badge = "新朋友 · 陌生人"
                confidence = max(master_score, best_friend_score, 0.0)

            # 归一化坐标百分比 (方便前端响应式渲染对焦框)
            left_pct = round((x1 / w) * 100, 2)
            top_pct = round((y1 / h) * 100, 2)
            width_pct = round(((x2 - x1) / w) * 100, 2)
            height_pct = round(((y2 - y1) / h) * 100, 2)

            parsed_targets.append({
                "target_index": i,
                "target_id": f"target_{i+1}",
                "target_title": f"目标 {i+1}",
                "identity": identity,
                "label": label,
                "display_badge": display_badge,
                "confidence": round(confidence, 2),
                "master_score": round(master_score, 3),
                "area": area,
                "bbox": [x1, y1, x2, y2],
                "bbox_pct": {
                    "left": left_pct,
                    "top": top_pct,
                    "width": width_pct,
                    "height": height_pct,
                },
            })

        # ======================================================
        # 多人选择性锁定逻辑
        # ======================================================
        locked_idx = 0
        if target_index is not None and 0 <= target_index < len(parsed_targets):
            # 用户手动选定目标（点击目镜按钮或语音切换）
            locked_idx = target_index
        else:
            # 策略一：若存在主人（相似度 >= 0.6），按主人最高匹配度锁定
            master_targets = [t for t in parsed_targets if t["identity"] == "master"]
            if master_targets:
                master_targets.sort(key=lambda x: x["master_score"], reverse=True)
                locked_idx = master_targets[0]["target_index"]
            else:
                # 策略二：全为陌生人或好友时，近场 C 位最大面积锁定
                sorted_by_area = sorted(parsed_targets, key=lambda x: x["area"], reverse=True)
                locked_idx = sorted_by_area[0]["target_index"]

        # 标记锁定态
        for t in parsed_targets:
            t["is_locked"] = t["target_index"] == locked_idx

        locked_target = parsed_targets[locked_idx]

        # 推荐语言反馈
        if locked_target["identity"] == "master":
            greeting_text = "主人，你回来啦！"
        elif locked_target["identity"] == "friend":
            greeting_text = f"检测到{locked_target['label']}在镜头前，你好呀！"
        else:
            greeting_text = "检测到新朋友在镜头前，你好呀！"

        with self._lock:
            self._last_detected_faces = parsed_targets
            self._last_locked_index = locked_idx

        logger.info(
            f"[Face] Detected {len(parsed_targets)} faces in {latency_ms}ms. "
            f"Locked target {locked_idx+1}: {locked_target['display_badge']}"
        )

        return {
            "success": True,
            "faces_count": len(parsed_targets),
            "latency_ms": latency_ms,
            "locked_index": locked_idx,
            "locked_target": locked_target,
            "targets": parsed_targets,
            "greeting_text": greeting_text,
        }

    def switch_to_next_target(self) -> dict[str, Any]:
        """语音/按钮指令：把视线焦点挪到旁边/下一个人的脸上"""
        with self._lock:
            faces = list(self._last_detected_faces)
            cur_idx = self._last_locked_index

        if not faces:
            return {"success": False, "error": "当前未检测到其他人脸目标"}

        if len(faces) == 1:
            return {
                "success": True,
                "message": "当前画面中只有一个人",
                "locked_target": faces[0],
                "greeting_text": f"当前画面只有一位朋友，是{faces[0]['label']}哦",
            }

        # 切换到下一个目标
        next_idx = (cur_idx + 1) % len(faces)
        for f in faces:
            f["is_locked"] = f["target_index"] == next_idx

        new_locked = faces[next_idx]
        with self._lock:
            self._last_locked_index = next_idx

        if new_locked["identity"] == "master":
            greeting = "这位是主人呀！"
        elif new_locked["identity"] == "friend":
            greeting = f"旁边这位是{new_locked['label']}朋友哦！"
        else:
            greeting = "旁边这位是新朋友呢，你好呀！"

        logger.info(f"[Face] Switched to target {next_idx+1}: {new_locked['display_badge']}")
        return {
            "success": True,
            "locked_index": next_idx,
            "locked_target": new_locked,
            "targets": faces,
            "greeting_text": greeting,
        }


# 全局单例
companion_face = CompanionFaceEngine()
