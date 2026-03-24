"""
噼哩噼哩 Pilipili-AutoVideo
视频生成模块 - Kling Omni / Kling v3 / Seedance 1.5

v2.0 改动：
- 新增 Kling Omni API（/v1/videos/omni-video）支持
  - 多镜头模式（multi_shot=true，最多6个分镜一次调用）
  - 多参考图（image_list，最多3个主体）
  - 首尾帧控制（first_frame_image + end_frame_image）
  - 文生视频（无参考图）
- 新增 shot_mode 自动判断逻辑
  - multi_ref：有固定人物 + 动作场景 → Omni 多参考生视频
  - first_end_frame：场景转换/运镜 → Omni 首尾帧生视频
  - t2v：纯风景/氛围 → Omni 文生视频
  - i2v：兼容旧版图生视频（默认回退）
- 保留 Seedance 1.5 作为备选引擎
"""

import os
import asyncio
import aiohttp
import json
import time
import jwt
import base64
from pathlib import Path
from typing import Optional, Literal
from datetime import datetime

from core.config import PilipiliConfig, get_config
from modules.llm import Scene


# ============================================================
# shot_mode 类型定义
# ============================================================

ShotMode = Literal["multi_ref", "first_end_frame", "t2v", "i2v"]


def auto_detect_shot_mode(scene: Scene) -> ShotMode:
    """
    根据分镜内容自动判断最优的生成模式

    判断规则：
    - 有角色参考图 + 有人物动作 → multi_ref（Omni 多参考生视频）
    - 有首尾帧关键词（转场/运镜/过渡）→ first_end_frame（Omni 首尾帧）
    - 纯风景/氛围/无人物 → t2v（Omni 文生视频）
    - 其他（默认）→ i2v（传统图生视频，兼容旧版）
    """
    # 如果 scene 已经有 shot_mode 字段，直接使用
    if hasattr(scene, "shot_mode") and scene.shot_mode:
        return scene.shot_mode

    prompt_lower = (scene.video_prompt + " " + scene.image_prompt + " " + " ".join(scene.style_tags)).lower()

    # 有角色参考图 → 优先使用 multi_ref
    if scene.reference_character:
        return "multi_ref"

    # 转场/运镜/过渡关键词 → first_end_frame
    transition_keywords = [
        "transition", "morph", "transform", "cut to", "fade to",
        "time lapse", "timelapse", "time-lapse", "dissolve",
        "转场", "过渡", "变换", "延时", "时光流逝"
    ]
    if any(kw in prompt_lower for kw in transition_keywords):
        return "first_end_frame"

    # 纯风景/氛围/无人物 → t2v
    landscape_keywords = [
        "landscape", "scenery", "nature", "sky", "ocean", "mountain",
        "forest", "sunset", "sunrise", "clouds", "aerial", "drone",
        "风景", "自然", "天空", "海洋", "山脉", "森林", "日落", "日出",
        "云彩", "航拍", "无人机", "空镜"
    ]
    person_keywords = [
        "person", "people", "man", "woman", "character", "figure",
        "人物", "人", "男", "女", "角色", "主角"
    ]
    has_landscape = any(kw in prompt_lower for kw in landscape_keywords)
    has_person = any(kw in prompt_lower for kw in person_keywords)

    if has_landscape and not has_person:
        return "t2v"

    # 默认回退到传统 i2v
    return "i2v"


# ============================================================
# 视频引擎路由逻辑
# ============================================================

def smart_route_engine(scene: Scene, default: str = "kling") -> str:
    """
    根据场景内容智能选择视频引擎

    规则：
    - 包含对话/口型同步关键词 → Seedance（原生音素级口型同步）
    - 包含多人/多角色场景 → Seedance（多主体一致性更强）
    - 包含动作/运动/体育 → Kling（动态能量更强）
    - 其他 → 使用默认引擎
    """
    seedance_keywords = [
        "talking", "speaking", "dialogue", "conversation", "lip sync",
        "multiple characters", "crowd", "group", "people talking",
        "interview", "narration", "说话", "对话", "多人", "人群"
    ]

    kling_keywords = [
        "action", "running", "jumping", "sports", "explosion", "fast",
        "dynamic", "energetic", "chase", "fight", "dance",
        "动作", "奔跑", "跳跃", "运动", "爆炸", "快速", "舞蹈"
    ]

    prompt_lower = (scene.video_prompt + " " + " ".join(scene.style_tags)).lower()

    seedance_score = sum(1 for kw in seedance_keywords if kw.lower() in prompt_lower)
    kling_score = sum(1 for kw in kling_keywords if kw.lower() in prompt_lower)

    if seedance_score > kling_score:
        return "seedance"
    elif kling_score > seedance_score:
        return "kling"
    else:
        return default


# ============================================================
# Kling JWT 认证
# ============================================================

def _generate_kling_jwt(api_key: str, api_secret: str) -> str:
    """生成 Kling API JWT Token（遵循官方文档，显式传 headers）"""
    jwt_headers = {
        "alg": "HS256",
        "typ": "JWT",
    }
    payload = {
        "iss": api_key,
        "exp": int(time.time()) + 1800,
        "nbf": int(time.time()) - 5,
    }
    return jwt.encode(payload, api_secret, algorithm="HS256", headers=jwt_headers)


def _image_to_base64(image_path: str) -> str:
    """将图片转为纯 base64 字符串（不含 data URI 前缀）"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ============================================================
# Kling Omni API（v2.0 新增）
# ============================================================

def _build_omni_prompt(
    scene: Scene,
    shot_mode: str,
    reference_images: Optional[list[str]] = None,
    image_index_offset: int = 0,
) -> tuple[str, list[str]]:
    """
    构建 Kling Omni 导演式提示词，使用 <<<image_N>>> 格式引用参考图

    Returns:
        (prompt_text, image_b64_list)
        - prompt_text: 包含 <<<image_1>>> 等占位符的提示词
        - image_b64_list: 对应的 base64 图片列表（按顺序）
    """
    image_b64_list = []
    image_refs = []  # 用于收集 <<<image_N>>> 引用

    # 收集参考图（角色参考优先）
    ref_paths = []
    if scene.character_refs:
        ref_paths.extend(scene.character_refs[:3])
    elif scene.reference_character and os.path.exists(scene.reference_character):
        ref_paths.append(scene.reference_character)
    elif reference_images:
        ref_paths.extend([p for p in reference_images[:3] if os.path.exists(p)])

    for i, ref_path in enumerate(ref_paths):
        idx = image_index_offset + i + 1
        image_b64_list.append(_image_to_base64(ref_path))
        image_refs.append(f"<<<image_{idx}>>>")

    # 构建导演式提示词
    # 格式：[全局背景] + [主体引用] + [时序动作] + [运镜]
    base_prompt = scene.video_prompt

    if image_refs and shot_mode == "multi_ref":
        # 导演式结构：将角色引用嵌入提示词
        char_ref_str = " and ".join(image_refs)
        prompt = (
            f"{base_prompt}. "
            f"Main character: {char_ref_str}. "
            f"Maintain character consistency throughout all shots."
        )
    else:
        prompt = base_prompt

    return prompt, image_b64_list


async def _submit_kling_omni(
    scenes: list[Scene],
    image_paths: dict[int, str],
    config: PilipiliConfig,
    session: aiohttp.ClientSession,
    reference_images: Optional[list[str]] = None,
) -> str:
    """
    提交 Kling Omni 多镜头任务，返回 task_id

    支持模式（Omni3 官方格式）：
    - multi_ref：多参考图生视频，prompt 中用 <<<image_1>>> 引用角色
    - first_end_frame：首尾帧控制（指定起止画面，AI 补中间动画）
    - t2v：纯文生视频（无参考图，适合风景/氛围镜头）
    - i2v：单图生视频（兼容旧版）

    Omni3 Prompt 设计原则（导演式结构）：
    [全局背景] + [主体引用 <<<image_1>>>] + [时序动作描述] + [运镜语言]

    Args:
        scenes: 分镜列表（最多6个）
        image_paths: {scene_id: keyframe_path}
        config: 配置
        session: aiohttp session
        reference_images: 全局角色参考图路径列表（可选）
    """
    api_key = config.video_gen.kling.api_key
    api_secret = config.video_gen.kling.api_secret

    if not api_key or not api_secret:
        raise ValueError("Kling API Key/Secret 未配置")

    token = _generate_kling_jwt(api_key, api_secret)

    # 全局参考图列表（所有分镜共享的 image_list）
    # Omni3 支持在顶层传入 image_list，所有分镜的 <<<image_N>>> 都引用这个列表
    global_image_list = []
    global_image_paths = []  # 用于追踪已加入的路径，避免重复

    # 收集所有分镜的参考图（去重）
    all_ref_paths = []
    if reference_images:
        all_ref_paths.extend([p for p in reference_images[:3] if os.path.exists(p)])
    for scene in scenes:
        if scene.character_refs:
            for p in scene.character_refs:
                if p and os.path.exists(p) and p not in all_ref_paths:
                    all_ref_paths.append(p)
        elif scene.reference_character and os.path.exists(scene.reference_character):
            if scene.reference_character not in all_ref_paths:
                all_ref_paths.append(scene.reference_character)

    for ref_path in all_ref_paths[:4]:  # Omni 最多4张参考图
        global_image_list.append(_image_to_base64(ref_path))
        global_image_paths.append(ref_path)

    # 构建 multi_prompt 列表
    multi_prompt = []
    for scene in scenes:
        shot_mode = auto_detect_shot_mode(scene)
        duration = 5 if scene.duration <= 7 else 10

        # 构建导演式提示词（含 <<<image_N>>> 引用）
        if global_image_list and shot_mode == "multi_ref":
            # 确定该分镜使用哪些参考图
            char_refs = []
            if scene.character_refs:
                for p in scene.character_refs[:2]:
                    if p in global_image_paths:
                        idx = global_image_paths.index(p) + 1
                        char_refs.append(f"<<<image_{idx}>>>")
            elif scene.reference_character and scene.reference_character in global_image_paths:
                idx = global_image_paths.index(scene.reference_character) + 1
                char_refs.append(f"<<<image_{idx}>>>")
            elif global_image_paths:
                # 默认使用第一张参考图
                char_refs.append("<<<image_1>>>")

            if char_refs:
                char_ref_str = " and ".join(char_refs)
                prompt = (
                    f"{scene.video_prompt}. "
                    f"Main character: {char_ref_str}. "
                    f"Maintain character consistency throughout."
                )
            else:
                prompt = scene.video_prompt
        else:
            prompt = scene.video_prompt

        shot_data = {
            "prompt": prompt,
            "duration": duration,
            "shot_mode": shot_mode,
        }

        # 根据 shot_mode 添加对应参数
        if shot_mode == "first_end_frame":
            # 首尾帧：用关键帧作为首帧
            # 核心逻辑：强制指定第1帧和最后一帧，AI 负责生成中间补帧动画
            kf_path = image_paths.get(scene.scene_id)
            if kf_path and os.path.exists(kf_path):
                shot_data["first_frame_image"] = _image_to_base64(kf_path)

        elif shot_mode in ("i2v", "multi_ref"):
            # 图生视频 / 多参考：用关键帧作为首帧
            kf_path = image_paths.get(scene.scene_id)
            if kf_path and os.path.exists(kf_path):
                shot_data["first_frame_image"] = _image_to_base64(kf_path)

        # t2v 模式：不传图片，纯文生视频

        multi_prompt.append(shot_data)

    # 确定分辨率
    quality = config.video_gen.kling.default_quality or "high"
    resolution = "1080p" if quality == "high" else "720p"

    payload = {
        "model_name": "kling-video-o1",  # Omni 专用模型
        "multi_shot": True,
        "shot_type": "intelligence",     # Omni3 智能分镜模式
        "multi_prompt": multi_prompt,
        "negative_prompt": "blurry, low quality, distorted, deformed, ugly, bad anatomy",
        "cfg_scale": 0.5,
        "mode": "std",
        "aspect_ratio": config.video_gen.kling.default_ratio or "16:9",
        "resolution": resolution,
    }

    # 如果有全局参考图，加入顶层 image_list
    if global_image_list:
        payload["image_list"] = global_image_list

    url = f"{config.video_gen.kling.base_url}/v1/videos/omni-video"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    async with session.post(url, json=payload, headers=headers) as resp:
        resp_text = await resp.text()
        try:
            result = json.loads(resp_text)
        except json.JSONDecodeError:
            raise RuntimeError(f"Kling Omni API 返回非 JSON 响应 (HTTP {resp.status}): {resp_text[:200]}")

    if result.get("code") != 0:
        raise RuntimeError(
            f"Kling Omni 任务提交失败 (code={result.get('code')}, msg={result.get('message')}): {result}"
        )

    return result["data"]["task_id"]


async def _poll_kling_omni_task(
    task_id: str,
    config: PilipiliConfig,
    session: aiohttp.ClientSession,
    timeout: int = 600,
    poll_interval: int = 5,
) -> list[str]:
    """
    轮询 Kling Omni 任务状态，返回视频 URL 列表（多镜头）

    Returns:
        list[str]: 按分镜顺序排列的视频 URL 列表
    """
    api_key = config.video_gen.kling.api_key
    api_secret = config.video_gen.kling.api_secret

    url = f"{config.video_gen.kling.base_url}/v1/videos/omni-video/{task_id}"
    start_time = time.time()

    while time.time() - start_time < timeout:
        token = _generate_kling_jwt(api_key, api_secret)
        headers = {"Authorization": f"Bearer {token}"}

        async with session.get(url, headers=headers) as resp:
            resp_text = await resp.text()
            try:
                result = json.loads(resp_text)
            except json.JSONDecodeError:
                raise RuntimeError(f"Kling Omni 轮询返回非 JSON 响应 (HTTP {resp.status}): {resp_text[:200]}")

        if result.get("code") != 0:
            raise RuntimeError(
                f"Kling Omni 任务查询失败 (code={result.get('code')}, msg={result.get('message')}): {result}"
            )

        status = result["data"]["task_status"]

        if status == "succeed":
            videos = result["data"]["task_result"]["videos"]
            if videos:
                return [v["url"] for v in videos]
            raise RuntimeError("Kling Omni 任务成功但无视频 URL")

        elif status == "failed":
            raise RuntimeError(f"Kling Omni 任务失败: {result['data'].get('task_status_msg', '未知错误')}")

        await asyncio.sleep(poll_interval)

    raise TimeoutError(f"Kling Omni 任务 {task_id} 超时（{timeout}s）")


# ============================================================
# Kling v3 API（旧版，单镜头图生视频，保留作为回退）
# ============================================================

async def _submit_kling_i2v(
    image_path: str,
    scene: Scene,
    config: PilipiliConfig,
    session: aiohttp.ClientSession,
) -> str:
    """提交 Kling v3 I2V 任务，返回 task_id（旧版接口，作为 Omni 回退）"""
    api_key = config.video_gen.kling.api_key
    api_secret = config.video_gen.kling.api_secret

    if not api_key or not api_secret:
        raise ValueError("Kling API Key/Secret 未配置")

    token = _generate_kling_jwt(api_key, api_secret)
    img_b64 = _image_to_base64(image_path)

    duration = 5 if scene.duration <= 7 else 10
    quality = config.video_gen.kling.default_quality or "high"
    resolution = "1080p" if quality == "high" else "720p"

    payload = {
        "model_name": config.video_gen.kling.model or "kling-v3",
        "image": img_b64,
        "prompt": scene.video_prompt,
        "negative_prompt": "blurry, low quality, distorted, deformed, ugly, bad anatomy",
        "cfg_scale": 0.5,
        "mode": "std",
        "duration": str(duration),
        "aspect_ratio": config.video_gen.kling.default_ratio or "16:9",
        "resolution": resolution,
    }

    url = f"{config.video_gen.kling.base_url}/v1/videos/image2video"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    async with session.post(url, json=payload, headers=headers) as resp:
        resp_text = await resp.text()
        try:
            result = json.loads(resp_text)
        except json.JSONDecodeError:
            raise RuntimeError(f"Kling API 返回非 JSON 响应 (HTTP {resp.status}): {resp_text[:200]}")

    if result.get("code") != 0:
        raise RuntimeError(f"Kling 任务提交失败 (code={result.get('code')}, msg={result.get('message')}): {result}")

    return result["data"]["task_id"]


async def _poll_kling_task(
    task_id: str,
    config: PilipiliConfig,
    session: aiohttp.ClientSession,
    timeout: int = 300,
    poll_interval: int = 5,
) -> str:
    """轮询 Kling v3 任务状态，返回视频 URL"""
    api_key = config.video_gen.kling.api_key
    api_secret = config.video_gen.kling.api_secret

    url = f"{config.video_gen.kling.base_url}/v1/videos/image2video/{task_id}"
    start_time = time.time()

    while time.time() - start_time < timeout:
        token = _generate_kling_jwt(api_key, api_secret)
        headers = {"Authorization": f"Bearer {token}"}

        async with session.get(url, headers=headers) as resp:
            resp_text = await resp.text()
            try:
                result = json.loads(resp_text)
            except json.JSONDecodeError:
                raise RuntimeError(f"Kling 轮询返回非 JSON 响应 (HTTP {resp.status}): {resp_text[:200]}")

        if result.get("code") != 0:
            raise RuntimeError(f"Kling 任务查询失败 (code={result.get('code')}, msg={result.get('message')}): {result}")

        status = result["data"]["task_status"]

        if status == "succeed":
            videos = result["data"]["task_result"]["videos"]
            if videos:
                return videos[0]["url"]
            raise RuntimeError("Kling 任务成功但无视频 URL")

        elif status == "failed":
            raise RuntimeError(f"Kling 任务失败: {result['data'].get('task_status_msg', '未知错误')}")

        await asyncio.sleep(poll_interval)

    raise TimeoutError(f"Kling 任务 {task_id} 超时（{timeout}s）")


# ============================================================
# Seedance 1.5 API
# ============================================================

async def _submit_seedance_i2v(
    image_path: str,
    scene: Scene,
    config: PilipiliConfig,
    session: aiohttp.ClientSession,
) -> str:
    """提交 Seedance I2V 任务，返回 task_id"""
    api_key = config.video_gen.seedance.api_key

    if not api_key:
        raise ValueError("Seedance (Volcengine) API Key 未配置")

    ext = Path(image_path).suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    mime_type = mime_map.get(ext, "image/jpeg")
    img_b64 = _image_to_base64(image_path)
    image_data_url = f"data:{mime_type};base64,{img_b64}"

    duration = 5 if scene.duration <= 7 else 10

    payload = {
        "model": config.video_gen.seedance.model or "doubao-seedance-1-5-pro-250528",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": image_data_url}
            },
            {
                "type": "text",
                "text": scene.video_prompt
            }
        ],
        "duration": duration,
        "ratio": config.video_gen.seedance.default_ratio or "16:9",
        "seed": -1,
    }

    url = f"{config.video_gen.seedance.base_url}/contents/generations/tasks"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with session.post(url, json=payload, headers=headers) as resp:
        resp_text = await resp.text()
        try:
            result = json.loads(resp_text)
        except json.JSONDecodeError:
            raise RuntimeError(f"Seedance API 返回非 JSON 响应 (HTTP {resp.status}): {resp_text[:200]}")

    if "id" not in result:
        raise RuntimeError(f"Seedance 任务提交失败: {result}")

    return result["id"]


async def _poll_seedance_task(
    task_id: str,
    config: PilipiliConfig,
    session: aiohttp.ClientSession,
    timeout: int = 300,
    poll_interval: int = 5,
) -> str:
    """轮询 Seedance 任务状态，返回视频 URL"""
    api_key = config.video_gen.seedance.api_key
    url = f"{config.video_gen.seedance.base_url}/contents/generations/tasks/{task_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    start_time = time.time()

    while time.time() - start_time < timeout:
        async with session.get(url, headers=headers) as resp:
            resp_text = await resp.text()
            try:
                result = json.loads(resp_text)
            except json.JSONDecodeError:
                raise RuntimeError(f"Seedance 轮询返回非 JSON 响应 (HTTP {resp.status}): {resp_text[:200]}")

        status = result.get("status", "")

        if status == "succeeded":
            content = result.get("content", [])
            for item in content:
                if item.get("type") == "video_url":
                    return item["video_url"]["url"]
            raise RuntimeError("Seedance 任务成功但无视频 URL")

        elif status == "failed":
            raise RuntimeError(f"Seedance 任务失败: {result.get('error', '未知错误')}")

        await asyncio.sleep(poll_interval)

    raise TimeoutError(f"Seedance 任务 {task_id} 超时（{timeout}s）")


# ============================================================
# 统一视频生成接口
# ============================================================

async def generate_video_clip(
    scene: Scene,
    image_path: str,
    output_dir: str,
    engine: Optional[str] = None,
    auto_route: bool = True,
    config: Optional[PilipiliConfig] = None,
    verbose: bool = False,
    reference_images: Optional[list[str]] = None,
) -> str:
    """
    为单个分镜生成视频片段（使用旧版 i2v 接口）

    Args:
        scene: 分镜场景对象
        image_path: 首帧关键图路径
        output_dir: 输出目录
        engine: 指定引擎 "kling" / "kling_omni" / "seedance"（可选）
        auto_route: 是否启用智能路由
        config: 配置对象
        verbose: 是否打印调试信息
        reference_images: 角色参考图路径列表（用于 Omni multi_ref 模式）

    Returns:
        本地视频文件路径
    """
    if config is None:
        config = get_config()

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"scene_{scene.scene_id:03d}_clip.mp4")

    # 断点续传
    if os.path.exists(output_path):
        if verbose:
            print(f"[VideoGen] Scene {scene.scene_id} 视频片段已存在，跳过")
        return output_path

    # 选择引擎
    if engine:
        selected_engine = engine
    elif auto_route:
        selected_engine = smart_route_engine(scene, config.video_gen.default_provider)
    else:
        selected_engine = config.video_gen.default_provider

    if verbose:
        shot_mode = auto_detect_shot_mode(scene)
        print(f"[VideoGen] Scene {scene.scene_id} 使用引擎: {selected_engine}, shot_mode: {shot_mode}")
        print(f"[VideoGen] 视频提示词: {scene.video_prompt[:80]}...")

    async with aiohttp.ClientSession() as session:
        if selected_engine in ("kling", "kling_omni"):
            # 单分镜也走 Omni 接口（更好的角色一致性）
            try:
                task_id = await _submit_kling_omni(
                    scenes=[scene],
                    image_paths={scene.scene_id: image_path},
                    config=config,
                    session=session,
                    reference_images=reference_images,
                )
                if verbose:
                    print(f"[VideoGen] Kling Omni 任务已提交: {task_id}")
                video_urls = await _poll_kling_omni_task(task_id, config, session)
                video_url = video_urls[0] if video_urls else None
                if not video_url:
                    raise RuntimeError("Kling Omni 返回空视频 URL")
            except Exception as omni_err:
                if verbose:
                    print(f"[VideoGen] Kling Omni 失败 ({omni_err})，回退到 v3 i2v 接口")
                # 回退到旧版 i2v
                task_id = await _submit_kling_i2v(image_path, scene, config, session)
                if verbose:
                    print(f"[VideoGen] Kling v3 任务已提交: {task_id}")
                video_url = await _poll_kling_task(task_id, config, session)

        elif selected_engine == "seedance":
            task_id = await _submit_seedance_i2v(image_path, scene, config, session)
            if verbose:
                print(f"[VideoGen] Seedance 任务已提交: {task_id}")
            video_url = await _poll_seedance_task(task_id, config, session)
        else:
            raise ValueError(f"不支持的视频引擎: {selected_engine}")

        # 下载视频
        if verbose:
            print(f"[VideoGen] Scene {scene.scene_id} 生成完成，下载中...")

        async with session.get(video_url) as resp:
            video_data = await resp.read()

    with open(output_path, "wb") as f:
        f.write(video_data)

    if verbose:
        print(f"[VideoGen] Scene {scene.scene_id} 视频已保存: {output_path}")

    return output_path


async def generate_video_clips_omni_batch(
    scenes: list[Scene],
    keyframe_paths: dict[int, str],
    output_dir: str,
    config: PilipiliConfig,
    reference_images: Optional[list[str]] = None,
    batch_size: int = 6,
    verbose: bool = False,
) -> dict[int, str]:
    """
    使用 Kling Omni 多镜头模式批量生成视频片段
    每次最多6个分镜一起提交，大幅减少 API 调用次数

    Args:
        scenes: 所有分镜列表
        keyframe_paths: {scene_id: keyframe_path}
        output_dir: 输出目录
        config: 配置
        reference_images: 全局角色参考图
        batch_size: 每批最多分镜数（Omni 最大6）
        verbose: 是否打印调试信息

    Returns:
        {scene_id: video_path} 字典
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # 检查哪些分镜已经有缓存
    pending_scenes = []
    for scene in scenes:
        output_path = os.path.join(output_dir, f"scene_{scene.scene_id:03d}_clip.mp4")
        if os.path.exists(output_path):
            results[scene.scene_id] = output_path
            if verbose:
                print(f"[VideoGen] Scene {scene.scene_id} 视频片段已存在，跳过")
        else:
            pending_scenes.append(scene)

    if not pending_scenes:
        return results

    # 按 batch_size 分批处理
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(pending_scenes), batch_size):
            batch = pending_scenes[i:i + batch_size]

            if verbose:
                scene_ids = [s.scene_id for s in batch]
                print(f"[VideoGen] Omni 批次 {i // batch_size + 1}: 分镜 {scene_ids}")

            try:
                task_id = await _submit_kling_omni(
                    scenes=batch,
                    image_paths=keyframe_paths,
                    config=config,
                    session=session,
                    reference_images=reference_images,
                )

                if verbose:
                    print(f"[VideoGen] Kling Omni 批次任务已提交: {task_id}")

                video_urls = await _poll_kling_omni_task(task_id, config, session)

                # 下载并保存每个分镜的视频
                for j, scene in enumerate(batch):
                    if j >= len(video_urls):
                        if verbose:
                            print(f"[VideoGen] Scene {scene.scene_id} 无对应视频 URL，跳过")
                        continue

                    video_url = video_urls[j]
                    output_path = os.path.join(output_dir, f"scene_{scene.scene_id:03d}_clip.mp4")

                    async with session.get(video_url) as resp:
                        video_data = await resp.read()

                    with open(output_path, "wb") as f:
                        f.write(video_data)

                    results[scene.scene_id] = output_path

                    if verbose:
                        print(f"[VideoGen] Scene {scene.scene_id} 视频已保存: {output_path}")

            except Exception as e:
                if verbose:
                    print(f"[VideoGen] Omni 批次失败 ({e})，逐个回退到 v3 i2v 接口")

                # 批次失败时逐个回退
                for scene in batch:
                    try:
                        kf_path = keyframe_paths.get(scene.scene_id)
                        if not kf_path or not os.path.exists(kf_path):
                            continue

                        task_id = await _submit_kling_i2v(kf_path, scene, config, session)
                        video_url = await _poll_kling_task(task_id, config, session)

                        output_path = os.path.join(output_dir, f"scene_{scene.scene_id:03d}_clip.mp4")
                        async with session.get(video_url) as resp:
                            video_data = await resp.read()
                        with open(output_path, "wb") as f:
                            f.write(video_data)

                        results[scene.scene_id] = output_path

                    except Exception as scene_err:
                        if verbose:
                            print(f"[VideoGen] Scene {scene.scene_id} 生成失败: {scene_err}")

    return results


async def generate_all_video_clips(
    scenes: list[Scene],
    keyframe_paths: dict[int, str],
    output_dir: str,
    engine: Optional[str] = None,
    auto_route: bool = True,
    config: Optional[PilipiliConfig] = None,
    max_concurrent: int = 3,
    verbose: bool = False,
    reference_images: Optional[list[str]] = None,
    use_omni_batch: bool = True,
) -> dict[int, str]:
    """
    生成所有分镜的视频片段

    v2.0 改动：
    - 默认使用 Kling Omni 批量模式（use_omni_batch=True）
    - 每批最多6个分镜，大幅减少 API 调用次数
    - 回退策略：Omni 失败 → 逐个 v3 i2v → Seedance

    Returns:
        {scene_id: video_path} 字典
    """
    if config is None:
        config = get_config()

    # 确定是否使用 Omni 批量模式
    selected_engine = engine or (config.video_gen.default_provider if not auto_route else None)
    use_omni = (
        use_omni_batch
        and selected_engine in (None, "kling", "kling_omni", "auto")
        and config.video_gen.kling.api_key
        and config.video_gen.kling.api_secret
    )

    if use_omni:
        if verbose:
            print(f"[VideoGen] 使用 Kling Omni 批量模式（每批最多6个分镜）")
        return await generate_video_clips_omni_batch(
            scenes=scenes,
            keyframe_paths=keyframe_paths,
            output_dir=output_dir,
            config=config,
            reference_images=reference_images,
            verbose=verbose,
        )

    # 回退到逐个生成模式（并发）
    if verbose:
        print(f"[VideoGen] 使用逐个生成模式（并发数: {max_concurrent}）")

    semaphore = asyncio.Semaphore(max_concurrent)
    results = {}

    async def _generate_with_semaphore(scene: Scene):
        async with semaphore:
            image_path = keyframe_paths.get(scene.scene_id)
            if not image_path or not os.path.exists(image_path):
                raise FileNotFoundError(f"Scene {scene.scene_id} 关键帧图片不存在: {image_path}")

            path = await generate_video_clip(
                scene=scene,
                image_path=image_path,
                output_dir=output_dir,
                engine=engine,
                auto_route=auto_route,
                config=config,
                verbose=verbose,
                reference_images=reference_images,
            )
            results[scene.scene_id] = path

    tasks = [_generate_with_semaphore(scene) for scene in scenes]
    await asyncio.gather(*tasks)

    return results


def generate_all_video_clips_sync(
    scenes: list[Scene],
    keyframe_paths: dict[int, str],
    output_dir: str,
    engine: Optional[str] = None,
    auto_route: bool = True,
    config: Optional[PilipiliConfig] = None,
    max_concurrent: int = 3,
    verbose: bool = False,
    reference_images: Optional[list[str]] = None,
    use_omni_batch: bool = True,
) -> dict[int, str]:
    """generate_all_video_clips 的同步版本"""
    return asyncio.run(generate_all_video_clips(
        scenes=scenes,
        keyframe_paths=keyframe_paths,
        output_dir=output_dir,
        engine=engine,
        auto_route=auto_route,
        config=config,
        max_concurrent=max_concurrent,
        verbose=verbose,
        reference_images=reference_images,
        use_omni_batch=use_omni_batch,
    ))
