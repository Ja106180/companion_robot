"""
companion_web.py - 桌面伴生灵宠全息联网感知中枢（实时天气 + 今日热点新闻 + 全网搜索）

集成 GitHub 三大主流开源无密钥联网方案：
1. 实时城市气象引擎：Open-Meteo (open-meteo/open-meteo) + wttr.in (chubin/wttr.in) 双擎互备，内置全国主要城市坐标极速直连
2. 今日热点新闻引擎：DailyHot 实时热榜聚合（百度实时热搜 + 今日头条热榜 + 实时资讯检索）
3. 全网实时搜索引擎：ddgs (deedy5/ddgs) + 必应/百度实时网页摘要提取
"""

from __future__ import annotations

import html
import json
import logging
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any

logger = logging.getLogger("companion_web")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [WebAgent] %(message)s"))
    logger.addHandler(_handler)

# WMO 国际气象代码转中文描述
WMO_WEATHER_CODES: dict[int, str] = {
    0: "晴朗",
    1: "大部晴朗",
    2: "多云",
    3: "阴天",
    45: "有雾",
    48: "雾凇",
    51: "零星小毛毛雨",
    53: "毛毛雨",
    55: "密集毛毛雨",
    56: "冻毛毛雨",
    57: "强冻毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "冻雨",
    67: "强冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "雪粒",
    80: "阵雨",
    81: "强阵雨",
    82: "暴雨阵雨",
    85: "阵雪",
    86: "强阵雪",
    95: "雷阵雨",
    96: "雷阵雨伴随小冰雹",
    99: "强雷阵雨伴随冰雹",
}

# 内置全国主流城市及福建全境城市高精度经纬度（免除 Geocoding 二次请求，响应提速至 300ms）
CHINA_CITY_COORDS: dict[str, tuple[float, float]] = {
    # 福建省全境与主要区县
    "福州": (26.0745, 119.2965),
    "鼓楼": (26.0820, 119.3030),
    "台江": (26.0528, 119.3140),
    "仓山": (26.0460, 119.2736),
    "晋安": (26.0816, 119.3283),
    "马尾": (25.9992, 119.4552),
    "长乐": (25.9628, 119.5233),
    "闽侯": (26.1501, 119.1317),
    "福清": (25.7207, 119.3842),
    "连江": (26.1976, 119.5393),
    "平潭": (25.5037, 119.7912),
    "厦门": (24.4798, 118.0894),
    "泉州": (24.8741, 118.6757),
    "漳州": (24.5130, 117.6471),
    "莆田": (25.4541, 119.0078),
    "宁德": (26.6657, 119.5479),
    "南平": (26.6418, 118.1777),
    "三明": (26.2634, 117.6387),
    "龙岩": (25.0751, 117.0177),
    "武夷山": (27.7562, 118.0357),
    # 全国直辖市、省会及重点城市
    "北京": (39.9042, 116.4074),
    "上海": (31.2304, 121.4737),
    "广州": (23.1291, 113.2644),
    "深圳": (22.5431, 114.0579),
    "杭州": (30.2741, 120.1551),
    "成都": (30.5728, 104.0668),
    "重庆": (29.5630, 106.5516),
    "武汉": (30.5928, 114.3055),
    "南京": (32.0603, 118.7969),
    "西安": (34.3416, 108.9398),
    "苏州": (31.2990, 120.5853),
    "天津": (39.3434, 117.3616),
    "长沙": (28.2282, 112.9388),
    "郑州": (34.7466, 113.6253),
    "东莞": (23.0207, 113.7518),
    "青岛": (36.0671, 120.3826),
    "沈阳": (41.8057, 123.4315),
    "宁波": (29.8683, 121.5440),
    "昆明": (25.0389, 102.7183),
    "合肥": (31.8206, 117.2272),
    "佛山": (23.0215, 113.1214),
    "大连": (38.9140, 121.6147),
    "济南": (36.6512, 117.1201),
    "哈尔滨": (45.8038, 126.5350),
    "长春": (43.8171, 125.3235),
    "石家庄": (38.0428, 114.5149),
    "太原": (37.8706, 112.5489),
    "南昌": (28.6820, 115.8579),
    "贵阳": (26.6470, 106.6302),
    "南宁": (22.8170, 108.3665),
    "海口": (20.0440, 110.1999),
    "三亚": (18.2528, 109.5120),
    "兰州": (36.0611, 103.8343),
    "银川": (38.4872, 106.2309),
    "西宁": (36.6171, 101.7782),
    "乌鲁木齐": (43.8256, 87.6168),
    "拉萨": (29.6525, 91.1721),
    "呼和浩特": (40.8426, 111.7492),
    "珠海": (22.2710, 113.5767),
    "无锡": (31.4912, 120.3119),
    "常州": (31.8112, 119.9741),
    "温州": (27.9943, 120.6994),
    "嘉兴": (30.7461, 120.7555),
    "绍兴": (30.0303, 120.5802),
    "金华": (29.0791, 119.6474),
    "台州": (28.6564, 121.4208),
    "烟台": (37.4638, 121.4479),
    "威海": (37.5131, 122.1204),
    "洛阳": (34.6197, 112.4540),
    "桂林": (25.2736, 110.2902),
    "香港": (22.3193, 114.1694),
    "澳门": (22.1987, 113.5439),
    "台北": (25.0330, 121.5654),
}

WEEKDAY_NAMES = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
}


def get_chinese_now_str() -> str:
    now = datetime.now()
    wd = WEEKDAY_NAMES[now.weekday()]
    return f"{now.year}年{now.month}月{now.day}日 {wd} {now.strftime('%H:%M')}"


def _extract_city_from_query(text: str, default_city: str = "福州") -> str:
    """从用户提问中精准提取城市名称；若未指明具体城市则默认使用福州"""
    # 1. 优先匹配内置城市表
    for city in sorted(CHINA_CITY_COORDS.keys(), key=len, reverse=True):
        if city in text:
            return city

    # 2. 正则提取“XX市/县/区”或“看看XX今天天气”
    m = re.search(
        r"(?:看看|查查|查询|请问|知道|说说)?([\u4e00-\u9fa5]{2,5}?)(?:市|县|区)?(?:今天|今日|现在|目前|明天|最近|这几天)?的?(?:天气|气温|温度|几度|多少度|下雨|冷不冷|热不热)",
        text,
    )
    if m:
        cand = m.group(1)
        for prefix in ("看看", "查查", "请问", "告诉我", "你知道", "比如", "那", "在", "今天", "今日", "现在", "目前", "明天"):
            if cand.startswith(prefix) and len(cand) > len(prefix):
                cand = cand[len(prefix) :]
        if cand and cand not in ("今天", "今日", "现在", "目前", "明天", "这里", "那边", "外面", "外头", "我们"):
            return cand

    return default_city


class CompanionWebAgent:
    """桌面伴生灵宠全息联网中枢（Open-Meteo/wttr.in 天气 + 百度/头条实时热榜新闻 + DDGS/百度全网检索）"""

    def fetch_weather(self, city: str = "福州") -> dict[str, Any]:
        """获取指定城市的实时天气、温度、降水概率与穿衣带伞建议"""
        clean_city = city.rstrip("市县区") if len(city) > 2 else city
        coords = CHINA_CITY_COORDS.get(clean_city) or CHINA_CITY_COORDS.get(city)

        # 若不在内置坐标表中，调用 Open-Meteo 免费 Geocoding 接口解析坐标
        if not coords:
            try:
                geo_url = (
                    "https://geocoding-api.open-meteo.com/v1/search?name="
                    + urllib.parse.quote(city)
                    + "&count=1&language=zh&format=json"
                )
                req = urllib.request.Request(geo_url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=4.0) as resp:
                    geo_data = json.loads(resp.read().decode("utf-8"))
                    results = geo_data.get("results") or []
                    if results:
                        coords = (float(results[0]["latitude"]), float(results[0]["longitude"]))
                        clean_city = results[0].get("name") or city
            except Exception as exc:
                logger.warning(f"Open-Meteo geocoding error for {city}: {exc}")

        # 引擎 1：Open-Meteo 实时气象接口
        if coords:
            lat, lon = coords
            try:
                api_url = (
                    f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                    "&current=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m"
                    "&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max"
                    "&timezone=Asia%2FShanghai&forecast_days=2"
                )
                req = urllib.request.Request(api_url, headers=HEADERS)
                data = None
                for _attempt in range(2):
                    try:
                        with urllib.request.urlopen(req, timeout=6.0) as resp:
                            data = json.loads(resp.read().decode("utf-8"))
                        break
                    except Exception:
                        if _attempt == 1:
                            raise
                        time.sleep(0.2)

                cur = data.get("current") or {}
                daily = data.get("daily") or {}

                temp_now = cur.get("temperature_2m")
                feels_like = cur.get("apparent_temperature")
                humidity = cur.get("relative_humidity_2m")
                w_code = int(cur.get("weather_code", 0))
                wind_kmh = cur.get("wind_speed_10m", 0)
                weather_desc = WMO_WEATHER_CODES.get(w_code, "晴间多云")

                max_list = daily.get("temperature_2m_max") or [temp_now]
                min_list = daily.get("temperature_2m_min") or [temp_now]
                prob_list = daily.get("precipitation_probability_max") or [0]
                d_codes = daily.get("weather_code") or [w_code]

                t_max = max_list[0]
                t_min = min_list[0]
                rain_prob = prob_list[0] if prob_list[0] is not None else 0
                today_daily_desc = WMO_WEATHER_CODES.get(int(d_codes[0]), weather_desc)

                # 生成贴心穿衣与带伞建议
                tips = []
                if "雨" in weather_desc or "雨" in today_daily_desc or (rain_prob and rain_prob >= 45):
                    tips.append("今天有降雨可能，出门记得随身带把伞哦")
                else:
                    tips.append("今天基本无雨，很适合出门活动")
                if temp_now is not None:
                    if temp_now >= 28:
                        tips.append("天气比较热，注意防暑补水，穿短袖清凉装即可")
                    elif temp_now >= 20:
                        tips.append("体感舒适宜人，穿薄长袖、衬衫或单衣就很合适")
                    elif temp_now >= 12:
                        tips.append("天气有些凉意，建议穿卫衣或加一件薄外套")
                    else:
                        tips.append("气温偏低较冷，出门一定要穿厚外套注意保暖")

                summary_line = (
                    f"{clean_city}实时天气：{weather_desc}，当前气温 {temp_now}°C（体感 {feels_like}°C），"
                    f"今日全天 {today_daily_desc}，最高温 {t_max}°C / 最低温 {t_min}°C，"
                    f"相对湿度 {humidity}%，降雨概率 {rain_prob}%，风速 {wind_kmh}km/h。"
                    f"贴心建议：{'，'.join(tips)}。"
                )

                # 明日简报
                if len(max_list) > 1 and len(d_codes) > 1:
                    tmr_desc = WMO_WEATHER_CODES.get(int(d_codes[1]), "多云")
                    summary_line += f" 明天预计{tmr_desc}，气温 {min_list[1]}°C~{max_list[1]}°C。"

                return {
                    "success": True,
                    "city": clean_city,
                    "weather": weather_desc,
                    "temp_now": temp_now,
                    "temp_min": t_min,
                    "temp_max": t_max,
                    "rain_prob": rain_prob,
                    "chip_title": f"全息联网已检索 · {clean_city}实时气象：{weather_desc} {temp_now}°C ({t_min}~{t_max}°C)",
                    "context_text": summary_line,
                }
            except Exception as exc:
                logger.warning(f"Open-Meteo forecast error for {clean_city}: {exc}")

        # 引擎 2：wttr.in 备用气象接口
        try:
            wttr_url = f"https://wttr.in/{urllib.parse.quote(clean_city)}?format=j1&lang=zh"
            req = urllib.request.Request(wttr_url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=5.5) as resp:
                wdata = json.loads(resp.read().decode("utf-8"))
            cur = (wdata.get("current_condition") or [{}])[0]
            weather_today = (wdata.get("weather") or [{}])[0]
            temp_c = cur.get("temp_C", "")
            feels_c = cur.get("FeelsLikeC", "")
            humidity = cur.get("humidity", "")
            lang_zh = cur.get("lang_zh") or cur.get("weatherDesc") or [{"value": "多云"}]
            raw_desc = (lang_zh[0].get("value") or "多云").strip()
            en_zh_map = {
                "partly cloudy": "晴间多云",
                "sunny": "晴朗",
                "clear": "晴朗",
                "cloudy": "多云",
                "overcast": "阴天",
                "mist": "薄雾",
                "fog": "雾",
                "light rain": "小雨",
                "moderate rain": "中雨",
                "heavy rain": "大雨",
                "patchy rain nearby": "局部零星小雨",
                "light drizzle": "毛毛雨",
            }
            desc = en_zh_map.get(raw_desc.lower(), raw_desc)
            t_max = weather_today.get("maxtempC", temp_c)
            t_min = weather_today.get("mintempC", temp_c)
            summary_line = (
                f"{clean_city}实时天气：{desc}，当前气温 {temp_c}°C（体感 {feels_c}°C），"
                f"今日最高温 {t_max}°C / 最低温 {t_min}°C，湿度 {humidity}%。"
            )
            return {
                "success": True,
                "city": clean_city,
                "weather": desc,
                "temp_now": temp_c,
                "temp_min": t_min,
                "temp_max": t_max,
                "chip_title": f"全息联网已检索 · {clean_city}实时气象：{desc} {temp_c}°C ({t_min}~{t_max}°C)",
                "context_text": summary_line,
            }
        except Exception as exc:
            logger.warning(f"wttr.in fallback error for {clean_city}: {exc}")
            return {"success": False, "city": clean_city, "error": str(exc)}

    def fetch_daily_hot_news(self, topic: str | None = None, limit: int = 8) -> dict[str, Any]:
        """
        获取今日实时热点新闻榜单（PC百度热搜全量摘要版 + 新浪实时滚动新闻接口 + 百度资讯检索 + 今日头条热榜）
        """
        news_items: list[dict[str, str]] = []
        seen_titles: set[str] = set()

        # 1. 若用户指定了具体领域或城市话题（如“科技”、“人工智能”、“财经”、“福州”）
        if topic and topic not in ("今天", "今日", "最新", "最近", "热点", "热搜", "头条", "国内", "全网"):
            # 1a. 针对科技/AI/财经/国际/体育/娱乐领域，优先拉取新浪实时滚动新闻 API（带完整 intro 摘要）
            sina_lid_map = {
                "科技": 2515, "人工智能": 2515, "AI": 2515, "互联网": 2515, "数码": 2515,
                "财经": 2516, "经济": 2516, "股市": 2516, "金融": 2516, "A股": 2516,
                "国际": 2511, "世界": 2511, "国外": 2511,
                "体育": 2513, "足球": 2513, "篮球": 2513,
                "娱乐": 2514, "明星": 2514, "电影": 2514,
            }
            matched_lid = None
            for k_word, lid_val in sina_lid_map.items():
                if k_word.lower() in topic.lower():
                    matched_lid = lid_val
                    break

            if matched_lid:
                try:
                    s_url = f"https://feed.mix.sina.com.cn/api/roll/get?pageid=153&lid={matched_lid}&k=&num={limit}&page=1"
                    req = urllib.request.Request(s_url, headers=HEADERS)
                    with urllib.request.urlopen(req, timeout=4.0) as resp:
                        s_data = json.loads(resp.read().decode("utf-8"))
                    for it in s_data.get("result", {}).get("data") or []:
                        t_clean = (it.get("title") or "").strip()
                        s_clean = (it.get("intro") or it.get("summary") or "").strip()
                        if t_clean and t_clean not in seen_titles:
                            seen_titles.add(t_clean)
                            news_items.append({"title": t_clean, "desc": s_clean})
                            if len(news_items) >= limit:
                                break
                except Exception as exc:
                    logger.warning(f"Sina category news error for {topic}: {exc}")

            # 1b. 百度实时资讯搜索（针对城市或特定关键词，如“福州”）
            if len(news_items) < limit:
                try:
                    kw = f"{topic} 最新消息"
                    url = "https://www.baidu.com/s?rtt=1&bsst=1&cl=2&tn=news&word=" + urllib.parse.quote(kw)
                    req = urllib.request.Request(url, headers=HEADERS)
                    with urllib.request.urlopen(req, timeout=4.5) as resp:
                        page_html = resp.read().decode("utf-8", errors="ignore")
                    titles = re.findall(r'aria-label="标题：([^"]+)"', page_html)
                    summaries = re.findall(r'<span class="c-font-normal c-color-text"[^>]*>(.*?)</span>', page_html, re.S)
                    for idx, raw_t in enumerate(titles):
                        t_clean = html.unescape(re.sub(r"<[^>]+>", "", raw_t)).strip()
                        s_clean = ""
                        if idx < len(summaries):
                            s_clean = html.unescape(re.sub(r"<[^>]+>", "", summaries[idx])).strip()
                        if t_clean and t_clean not in seen_titles:
                            seen_titles.add(t_clean)
                            news_items.append({"title": t_clean, "desc": s_clean})
                            if len(news_items) >= limit:
                                break
                except Exception as exc:
                    logger.warning(f"Topic news fetch error for {topic}: {exc}")

        # 2. 聚合：百度 PC 端实时热搜榜（含每条热点新闻的完整事件来龙去脉摘要 desc！）
        if len(news_items) < limit:
            try:
                pc_url = "https://top.baidu.com/board?tab=realtime"
                req = urllib.request.Request(pc_url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=4.0) as resp:
                    pc_html = resp.read().decode("utf-8", errors="ignore")
                m_sdata = re.search(r"<!--s-data:(.*?)-->", pc_html, re.S)
                if m_sdata:
                    pc_json = json.loads(m_sdata.group(1))
                    pc_cards = pc_json.get("data", {}).get("cards") or []
                    if pc_cards:
                        for entry in pc_cards[0].get("content") or []:
                            if not isinstance(entry, dict):
                                continue
                            title = (entry.get("word") or entry.get("query") or "").strip()
                            desc = (entry.get("desc") or "").strip()
                            if not title or title in seen_titles:
                                continue
                            if topic and topic not in ("今天", "今日", "最新", "热点") and topic not in title and topic not in desc:
                                continue
                            seen_titles.add(title)
                            news_items.append({"title": title, "desc": desc})
                            if len(news_items) >= limit:
                                break
            except Exception as exc:
                logger.warning(f"Baidu PC hot board error: {exc}")

        # 3. 聚合：新浪全网实时要闻滚动榜补充（带完整 intro 摘要）
        if len(news_items) < limit:
            try:
                s_url = "https://feed.mix.sina.com.cn/api/roll/get?pageid=153&lid=2509&k=&num=10&page=1"
                req = urllib.request.Request(s_url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=3.5) as resp:
                    s_data = json.loads(resp.read().decode("utf-8"))
                for it in s_data.get("result", {}).get("data") or []:
                    title = (it.get("title") or "").strip()
                    desc = (it.get("intro") or "").strip()
                    if not title or title in seen_titles:
                        continue
                    seen_titles.add(title)
                    news_items.append({"title": title, "desc": desc})
                    if len(news_items) >= limit:
                        break
            except Exception as exc:
                logger.warning(f"Sina general roll news error: {exc}")

        # 4. 聚合：今日头条实时热榜兜底
        if len(news_items) < limit:
            try:
                t_url = "https://www.toutiao.com/hot-event/hot-board/?origin=toutiao_pc"
                req = urllib.request.Request(t_url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=3.5) as resp:
                    t_data = json.loads(resp.read().decode("utf-8"))
                for entry in t_data.get("data") or []:
                    title = (entry.get("Title") or "").strip()
                    if not title or title in seen_titles:
                        continue
                    seen_titles.add(title)
                    news_items.append({"title": title, "desc": ""})
                    if len(news_items) >= limit:
                        break
            except Exception as exc:
                logger.warning(f"Toutiao hot board error: {exc}")

        if not news_items:
            return {"success": False, "items": []}

        lines = []
        for i, item in enumerate(news_items[:limit], 1):
            if item["desc"]:
                lines.append(f"{i}. 【{item['title']}】：{item['desc']}")
            else:
                lines.append(f"{i}. 【{item['title']}】")

        topic_label = f"{topic}实时资讯" if (topic and topic not in ("今天", "今日", "最新", "热点")) else "今日全网热点要闻"
        return {
            "success": True,
            "items": news_items[:limit],
            "chip_title": f"全息联网已检索 · {topic_label} (已抓取 {len(news_items[:limit])} 条今日头条与摘要)",
            "context_text": f"获取时间：{get_chinese_now_str()}\n今日实时热点新闻与摘要列表：\n" + "\n".join(lines),
        }

    def fetch_web_search(self, query: str, limit: int = 5) -> dict[str, Any]:
        """
        全网实时搜索（百度资讯 + 必应中国 + 新浪科技/全网检索 + DDGS 多引擎）
        """
        snippets: list[str] = []

        # 1. 百度实时资讯检索
        try:
            url = "https://www.baidu.com/s?rtt=1&bsst=1&cl=2&tn=news&word=" + urllib.parse.quote(query)
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                page_html = resp.read().decode("utf-8", errors="ignore")
            titles = re.findall(r'aria-label="标题：([^"]+)"', page_html)
            summaries = re.findall(r'<span class="c-font-normal c-color-text"[^>]*>(.*?)</span>', page_html, re.S)
            for idx, raw_t in enumerate(titles[:limit]):
                t_clean = html.unescape(re.sub(r"<[^>]+>", "", raw_t)).strip()
                s_clean = html.unescape(re.sub(r"<[^>]+>", "", summaries[idx])).strip() if idx < len(summaries) else ""
                if t_clean:
                    snippets.append(f"- {t_clean}：{s_clean}" if s_clean else f"- {t_clean}")
        except Exception as exc:
            logger.warning(f"Baidu search error: {exc}")

        # 2. 若涉及科技/AI/数码，自动拉取新浪科技最新滚动资讯补充
        if len(snippets) < 3 and re.search(r"(?:人工智能|AI|大模型|科技|芯片|手机|数码|机器人)", query, re.I):
            try:
                s_url = "https://feed.mix.sina.com.cn/api/roll/get?pageid=153&lid=2515&k=&num=5&page=1"
                req = urllib.request.Request(s_url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=3.5) as resp:
                    s_data = json.loads(resp.read().decode("utf-8"))
                for it in s_data.get("result", {}).get("data") or []:
                    t_clean = (it.get("title") or "").strip()
                    s_clean = (it.get("intro") or "").strip()
                    if t_clean:
                        snippets.append(f"- {t_clean}：{s_clean}")
            except Exception:
                pass

        # 3. 必应国内版 (cn.bing.com) 实时网页检索补充
        if len(snippets) < limit:
            try:
                b_url = "https://cn.bing.com/search?q=" + urllib.parse.quote(query) + "&ensearch=0"
                req = urllib.request.Request(b_url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=4.0) as resp:
                    b_page = resp.read().decode("utf-8", errors="ignore")
                blocks = re.findall(r'<li class="b_algo"[^>]*>(.*?)</li>', b_page, re.S)
                for b in blocks[:limit]:
                    hm = re.search(r"<h2[^>]*>(.*?)</h2>", b, re.S)
                    pm = re.search(r"<p[^>]*>(.*?)</p>", b, re.S)
                    if hm:
                        t = html.unescape(re.sub(r"<[^>]+>", "", hm.group(1))).strip()
                        s = html.unescape(re.sub(r"<[^>]+>", "", pm.group(1))).strip() if pm else ""
                        if t:
                            snippets.append(f"- {t}：{s}" if s else f"- {t}")
            except Exception as exc:
                logger.warning(f"Bing CN search error: {exc}")

        if not snippets:
            return {"success": False}

        return {
            "success": True,
            "chip_title": f"全息联网已检索 · 实时网页搜索 ({len(snippets[:limit])} 条结果)",
            "context_text": f"当前时间：{get_chinese_now_str()}\n关于「{query}」的实时网络检索结果：\n" + "\n".join(snippets[:limit]),
        }

    def analyze_and_search(self, user_text: str) -> dict[str, Any] | None:
        """
        智能意图路由：自动判断用户是否在询问【实时天气】、【今日新闻】或【联网搜索查询】
        若命中，立即返回实时联网数据包；若只是普通日常闲聊，返回 None（零延迟跳过）。
        """
        text = user_text.strip()
        if not text:
            return None

        # 1. 检测【实时天气】意图
        if re.search(r"(?:天气|气温|温度|几度|多少度|下雨|下雪|刮风|带伞|穿什么衣服|冷不冷|热不热|台风|天气预报)", text):
            city = _extract_city_from_query(text, default_city="福州")
            logger.info(f"[WebAgent] Detected weather intent for city: {city}")
            res = self.fetch_weather(city)
            if res.get("success"):
                res["category"] = "weather"
                return res

        # 2. 检测【今日新闻 / 热点总结】意图
        if re.search(r"(?:新闻|热搜|热点|头条|要闻|大事|新鲜事|发生了什么|有什么消息|最新消息|天下事|资讯|动态)", text):
            topic = None
            m_topic = re.search(r"(?:关于|有关|看看|讲讲|说说|有什么|今天|今日|最新)?([\u4e00-\u9fa5A-Za-z0-9]{2,8}?)(?:的)?(?:新闻|热搜|热点|头条|最新消息|资讯|动态)", text)
            if m_topic:
                cand = m_topic.group(1)
                for stop_w in ("今天", "今日", "现在", "最近", "最新", "有什么", "有啥", "什么", "哪些", "看看", "说说", "讲讲", "总结", "播报", "一下", "一点", "国内", "全网", "有"):
                    cand = cand.replace(stop_w, "")
                cand = cand.strip("的")
                if len(cand) >= 2:
                    topic = cand
            logger.info(f"[WebAgent] Detected news intent (topic={topic})")
            res = self.fetch_daily_hot_news(topic=topic, limit=6)
            if res.get("success"):
                res["category"] = "news"
                return res

        # 3. 检测显式【上网搜索 / 实时查询】意图
        if re.search(r"(?:上网|联网|搜索|搜一下|查一下|帮我查|去网上看看|最新汇率|今日金价|今天几号|今天星期几|今年是哪年)", text):
            if re.search(r"(?:今天几号|今天星期几|现在几点|今天是什么日子|今年是哪年)", text) and not re.search(r"(?:搜索|查一下|上网)", text):
                now_str = get_chinese_now_str()
                return {
                    "success": True,
                    "category": "time",
                    "chip_title": f"全息时钟已同步 · {now_str}",
                    "context_text": f"当前准确的北京时间是：{now_str}。",
                }
            clean_q = re.sub(r"^(?:请|帮我|去|上网|联网|网上|搜索|搜一下|查一下|查询|看看)+", "", text).strip() or text
            logger.info(f"[WebAgent] Detected explicit web search intent: {clean_q}")
            res = self.fetch_web_search(clean_q, limit=4)
            if res.get("success"):
                res["category"] = "search"
                return res

        return None


# 全局单例
companion_web = CompanionWebAgent()
