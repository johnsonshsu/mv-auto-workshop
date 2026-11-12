"""歌詞的資料結構、解析與轉換。

lyrics.json 的結構：
    {"title": "...", "language": "zh-TW",
     "sections": [{"type": "verse", "lines": ["...", "..."]}, ...]}
lines 為空代表純音樂段落。
"""
from __future__ import annotations

import re

from app.core.models import UserFacingError

SECTION_TYPES = ["intro", "verse", "prechorus", "chorus", "bridge", "outro"]

SECTION_LABELS = {
    "intro": "前奏",
    "verse": "主歌",
    "prechorus": "導歌",
    "chorus": "副歌",
    "bridge": "過渡",
    "outro": "尾奏",
}

# 人工輸入時接受的寫法（比對前會轉小寫並去掉空白、連字號）
_ALIASES = {
    "intro": ["前奏", "intro"],
    "verse": ["主歌", "verse"],
    "prechorus": ["導歌", "預副歌", "prechorus"],
    "chorus": ["副歌", "chorus", "hook"],
    "bridge": ["過渡", "橋段", "bridge"],
    "outro": ["尾奏", "尾聲", "outro"],
}
_ALIAS_TO_TYPE = {alias: t for t, aliases in _ALIASES.items() for alias in aliases}

# [主歌]、【副歌 2】、［Verse 1］、(Bridge)
_BRACKETED = re.compile(r"^[\[【［(（]\s*(.+?)\s*[\]】］)）]\s*[:：]?$")
# 沒有括號的寫法只接受中文標記：「主歌」、「副歌2：」
_BARE = re.compile(r"^(前奏|主歌|導歌|預副歌|副歌|過渡|橋段|尾奏|尾聲)\s*\d*\s*[:：]?$")

# 歌詞中不唱的符號；清理時換成空白
_PUNCTUATION = re.compile(r"[，。！？、；：,.!?;:「」『』“”\"…—~～·]+")


# 只存在於簡體的常用字（不含「后、里、云、只、台」這類繁體也會用到的字）。
# 用來抓出 LLM 混入的簡體字；不是完整字表，只求不誤判。
_SIMPLIFIED_ONLY = frozenset(
    "这们说时会对发现过还开关东车门问间长见风飞马鸟鱼语话谁请让认记读写书买卖钱银铁钟错听厅压严业专两为举义乌乐习乡"
    "亏亚产亲亿仅从仓仪众优伙传伤伦伪债倾偿储儿兰兴养兽军农决况冻净减凤凭击刘则刚创删别剂剑剧劝办务动励劲劳势华协单"
    "卢卫厉县参双变叙号叹吓吗吨启员呜响哑唤喷团园围图圆圣场坏块坚坛垒垦垫墙壮声壳处备复够头夺奋奖妇妈娱婴孙学宝实宠"
    "审宪宽宾寻导寿将尘尝层岁岂岗岛岭峡币帅师帘带帮库应庙庞废庆张弯弹强归录彻径忆忧怀态怜总恋恳恶悬惊惧惨惩愤愿懒戏"
    "战执扩扫扬扰抚抢护报担拟拥择挡挤挥损捡换摄摆敌数断无旧显晓晕杀杂条来杨构枪枫标样树桥档检楼欢欧毕气汉泼泽洁浅浆"
    "浇浊测济浏浑浓涛润涩渊渐渔渗湾湿溃满滚潜灭灵灾灿炉炼烁烂烛烦烧热爱爷牵独狭狮猎猪猫玛环现琐电畅疗疯皱盏盐监盖盘"
    "矫矿码砖础硕确礼祷祸离积称稳穷窃窍窝竖竞笔笼筛签简篮类粮纠红纤约级纪纯纱纲纳纵纷纸纹纺线练组细织终绍经绑绒结绕"
    "绘给绝统继绩绪续绳维绵综绿缓缔编缘缚缝缠缩绣罗罚罢翘耸聋职联聪肃肠肤胁胶脑脸腻舰舱艰艳节苏荡荣药莲莹营萧萨蓝虑"
    "虽虾蚀蚁蚂补装观规觉览订讨议讯训讲论访证评识诉词译试诗诚该详诱诵调谅谈谊谋谎谐谜谢谣谦谨谱贡财责贤败货质贩贪贫"
    "购贯贱贴贵贷贸费贺贼资赌赏赐赔赖赚赛赞赠赶趋跃踊踪躯轨转轮软轰轻载较辅辆辈辉辑输辩边辽达迁迈运进远违连迟选逊递"
    "逻遗邓邮邻郑酱释钓钞钢钥钩钮铃铅铜铝铲铺链销锁锅锈锋锐锦键锯镇镜闪闭闯闲闷闹闻阁阅队阳阴阵阶际陆陈险随隐难雾顶"
    "项顺须顽顾顿颂预领频颗题颜额飘饥饭饮饰饱饲饺饼馆驰驱驶驻驾验骂骄骑骗骚骤鲁鲜鸡鸣鸭鸽鹅鹰麦齐齿龙龟视")


def find_simplified(text: str) -> list[str]:
    """回傳 text 中出現的簡體字（不重複，依出現順序）。"""
    return list(dict.fromkeys(ch for ch in text if ch in _SIMPLIFIED_ONLY))


_converter = None


def to_traditional(text: str) -> str:
    """用 OpenCC 把簡體字轉成繁體（s2tw：只換字，不改用詞，字數不變）。沒安裝 OpenCC 時原樣回傳。"""
    global _converter
    if _converter is None:
        try:
            from opencc import OpenCC
            _converter = OpenCC("s2tw").convert
        except ImportError:
            _converter = lambda value: value          # noqa: E731
    return _converter(text)


class LyricsFormatError(UserFacingError):
    """人工輸入的歌詞格式有誤；problems 逐項指出是哪一行。"""

    def __init__(self, problems: list[str]):
        super().__init__("歌詞格式有誤：" + "；".join(problems),
                         "請用 [主歌]、[副歌]、[過渡] 或 [Verse]、[Chorus]、[Bridge] 標示段落")
        self.problems = problems


def marker_type(line: str) -> str | None:
    """line 是段落標記時回傳段落類型；是未知標記回傳 ""；不是標記回傳 None。"""
    bare = _BARE.match(line)
    if bare:
        return _ALIAS_TO_TYPE[bare.group(1)]
    bracketed = _BRACKETED.match(line)
    if not bracketed:
        return None
    name = re.sub(r"[\s\-_]+|\d+", "", bracketed.group(1)).lower()
    if name in _ALIAS_TO_TYPE:
        return _ALIAS_TO_TYPE[name]
    # 圓括號包住的整行可能只是歌詞（例如和聲），不當成寫錯的標記
    return None if line[0] in "(（" else ""


def parse_lyrics_text(text: str, title: str = "", language: str = "zh-TW") -> dict:
    """解析帶段落標記的歌詞文字。中英文標記都接受。"""
    sections: list[dict] = []
    problems: list[str] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        kind = marker_type(line)
        if kind == "":
            problems.append(f"第 {number} 行：看不懂的段落標記「{line}」")
        elif kind is not None:
            sections.append({"type": kind, "lines": []})
        elif not sections:
            problems.append(f"第 {number} 行：「{line[:12]}」前面沒有段落標記")
        else:
            sections[-1]["lines"].append(line)
    if not problems and not any(s["lines"] for s in sections):
        problems.append("沒有任何歌詞")
    if problems:
        raise LyricsFormatError(problems)
    return {"title": title, "language": language, "sections": sections}


def to_editor_text(lyrics: dict) -> str:
    """轉成歌詞頁編輯器顯示的文字（中文標記）。"""
    blocks = []
    for section in lyrics.get("sections", []):
        blocks.append("\n".join([f"[{SECTION_LABELS[section['type']]}]", *section["lines"]]))
    return "\n\n".join(blocks)


def lyric_lines(lyrics: dict) -> list[str]:
    """所有要唱的行，依序；行號（從 1 起）就是字幕編號。"""
    return [line for section in lyrics.get("sections", []) for line in section["lines"]]


def to_plain_text(lyrics: dict) -> str:
    """lyrics.txt 的內容：一行一句，供字幕對齊。"""
    return "\n".join(lyric_lines(lyrics)) + "\n"


def to_yue2(lyrics: dict, tag_map: dict[str, str]) -> str:
    """轉成 YuE2 歌詞格式。歌詞欄只放要唱的字，所以沒有歌詞的段落不輸出。"""
    blocks = []
    for section in lyrics.get("sections", []):
        if not section["lines"]:
            continue
        tag = tag_map.get(section["type"], "Verse")
        blocks.append("\n".join([f"[{tag}]", *section["lines"]]))
    return "\n\n".join(blocks)


def clean_line(line: str) -> str:
    """去掉不唱的標點與括號註解，多個空白併成一個。"""
    line = re.sub(r"[（(][^）)]*[）)]", "", line)
    return re.sub(r"\s+", " ", _PUNCTUATION.sub(" ", line)).strip()


def count_chars(line: str) -> int:
    """演唱字數：中日韓文字一字算一個，連續的英數字算一個。"""
    cjk = len(re.findall(r"[㐀-鿿豈-﫿぀-ヿ]", line))
    words = len(re.findall(r"[A-Za-z0-9']+", line))
    return cjk + words


def estimate_seconds(lyrics: dict, seconds_per_line: float, seconds_per_gap: float) -> float:
    """粗估歌曲長度：每行固定秒數，每個段落交界加一段間奏。"""
    sections = lyrics.get("sections", [])
    lines = sum(len(s["lines"]) for s in sections)
    instrumental = sum(1 for s in sections if not s["lines"])
    gaps = max(len(sections) - 1, 0) + instrumental
    return lines * seconds_per_line + gaps * seconds_per_gap


DEFAULT_BPM = 96


def bpm_from_text(text: str, default: int = DEFAULT_BPM) -> int:
    """從曲風提示或風格描述裡找出速度（例如 "around 72 BPM"）；找不到就用預設值。"""
    found = re.search(r"(\d{2,3})\s*BPM", text or "", flags=re.IGNORECASE)
    return min(max(int(found.group(1)), 50), 200) if found else default


def song_seconds(line_count: int, section_count: int, bpm: int) -> float:
    """估計唱完這些歌詞需要多久。

    一行歌詞大約兩小節（8 拍），段落之間的過門再算一行的長度，前奏約 10 秒、尾奏約 6 秒。
    與實機結果對照：22 行、108 BPM 估 136 秒（實際 140–150）；8 行、72 BPM 估 76 秒（45 秒唱不完）。
    """
    per_line = 480.0 / max(bpm, 1)
    return 10.0 + line_count * per_line + max(section_count - 1, 0) * per_line + 6.0


def lyrics_seconds(lyrics: dict, bpm: int) -> float:
    sections = [s for s in lyrics.get("sections", []) if s["lines"]]
    return song_seconds(sum(len(s["lines"]) for s in sections), len(sections), bpm)