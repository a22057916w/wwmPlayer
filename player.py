# midi_piano_game_player.py
# pip install mido pynput
#
# 功能重點：
# - 變速(set_tempo)完整處理：tick->sec 分段積分 + 絕對時間排程（不累積漂移）
# - RH/LH 選擇：有 Right/Left 名稱就用；否則用「音高較高」的 track 當 RH（p95）
# - 音域策略：RH 全保留（超出就八度折回），LH 太低就丟（可用 --left-drop-below 調）
# - 和弦策略：
#   --chord-mode onegroup   : 解法1（同一拍只保留一種 modifier 群）
#   --chord-mode arpeggio   : 解法3（把不同 modifier 群極短刷出來，調到人耳分不出）
#   --chord-mode raw        : 不處理（不建議）
# - arpeggio 加速參數：
#   --arp-gap-ms      : 群與群間隔（建議 0~2）
#   --arp-hold-ms     : 每群按住多久（建議 8~12）
#   --arp-max-groups  : 同一拍最多刷幾群（建議 2~3）
#
# 注意：要讓遊戲吃到按鍵，通常要用系統管理員權限執行 CMD/PowerShell。

import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import mido
from pynput.keyboard import Controller, Key


# =========================
# 你遊戲的鍵位映射（固定）
# =========================
MID_KEYS  = {"1": "a", "2": "s", "3": "d", "4": "f", "5": "g", "6": "h", "7": "j"}  # 中音
HIGH_KEYS = {"1": "q", "2": "w", "3": "e", "4": "r", "5": "t", "6": "y", "7": "u"}  # 高音
LOW_KEYS  = {"1": "z", "2": "x", "3": "c", "4": "v", "5": "b", "6": "n", "7": "m"}  # 低音

# 12 半音 -> 你的簡譜符號（以 pitch class 表示）
PC_TO_TOKEN = {
    0:  "1",   # C
    1:  "#1",  # C#
    2:  "2",   # D
    3:  "b3",  # Eb
    4:  "3",   # E
    5:  "4",   # F
    6:  "#4",  # F#
    7:  "5",   # G
    8:  "#5",  # G#
    9:  "6",   # A
    10: "b7",  # Bb
    11: "7",   # B
}

# 你的遊戲：升半音用 Shift、降半音用 Ctrl
ACC_SHARP = {"#1", "#4", "#5"}  # Shift
ACC_FLAT  = {"b3", "b7"}        # Ctrl

# =========================
# 你遊戲 36 鍵可覆蓋音域（可自行調整）
# 低音列: C3..B3 = 48..59
# 中音列: C4..B4 = 60..71
# 高音列: C5..B5 = 72..83
# =========================
LOW_MIN, LOW_MAX   = 48, 59
MID_MIN, MID_MAX   = 60, 71
HIGH_MIN, HIGH_MAX = 72, 83
PLAY_MAX = HIGH_MAX  # 上限固定 B5=83

kbd = Controller()


# =========================
# MIDI 事件結構
# =========================
@dataclass
class NoteSpan:
    start_tick: int
    end_tick: int
    note: int
    vel: int
    channel: int


@dataclass
class TempoChange:
    abs_tick: int
    tempo_us_per_beat: int  # microseconds per beat


@dataclass
class ScheduledNote:
    t_sec: float
    dur_sec: float
    note: int
    hand: str  # "RH"/"LH"


# =========================
# Tempo map（變速）處理
# =========================
def build_tempo_map(mid: mido.MidiFile) -> List[TempoChange]:
    """
    收集所有 set_tempo（常在 track0），排序後去重。
    若 tick=0 沒 tempo，補預設 120BPM (500000 us/beat)。
    """
    changes: List[TempoChange] = []

    for tr in mid.tracks:
        abs_tick = 0
        for msg in tr:
            abs_tick += msg.time
            if msg.type == "set_tempo":
                changes.append(TempoChange(abs_tick=abs_tick, tempo_us_per_beat=msg.tempo))

    changes.sort(key=lambda x: x.abs_tick)

    if not changes or changes[0].abs_tick != 0:
        changes.insert(0, TempoChange(abs_tick=0, tempo_us_per_beat=500000))

    # 同 tick 多個 tempo，只保留最後一個
    dedup: List[TempoChange] = []
    for c in changes:
        if dedup and dedup[-1].abs_tick == c.abs_tick:
            dedup[-1] = c
        else:
            dedup.append(c)
    return dedup


def tick_to_seconds(abs_tick: int, ticks_per_beat: int, tempo_map: List[TempoChange]) -> float:
    """
    分段積分：abs_tick -> 秒（真正吃 tempo map）
      delta_beat = delta_tick / ticks_per_beat
      delta_sec  = delta_beat * (tempo_us_per_beat / 1e6)
    """
    total = 0.0
    for i in range(len(tempo_map)):
        t0 = tempo_map[i].abs_tick
        tempo = tempo_map[i].tempo_us_per_beat
        t1 = tempo_map[i + 1].abs_tick if i + 1 < len(tempo_map) else abs_tick

        if abs_tick <= t0:
            break

        seg_end = min(abs_tick, t1)
        if seg_end > t0:
            delta_tick = seg_end - t0
            total += (delta_tick / ticks_per_beat) * (tempo / 1_000_000.0)

        if abs_tick < t1:
            break

    return total


# =========================
# Track / NoteSpan 抽取 + 分 RH/LH
# =========================
def get_track_name(track: mido.MidiTrack) -> Optional[str]:
    for msg in track:
        if msg.type == "track_name":
            return msg.name
    return None


def extract_note_spans(track: mido.MidiTrack) -> List[NoteSpan]:
    """
    note_on vel>0 = 按下
    note_off or note_on vel=0 = 放開
    """
    abs_tick = 0
    active: Dict[Tuple[int, int], Tuple[int, int]] = {}  # (ch,note)->(start_tick, vel)
    spans: List[NoteSpan] = []

    for msg in track:
        abs_tick += msg.time
        if msg.type not in ("note_on", "note_off"):
            continue

        ch = getattr(msg, "channel", 0)
        note = msg.note

        if msg.type == "note_on" and msg.velocity > 0:
            active[(ch, note)] = (abs_tick, msg.velocity)
        else:
            key = (ch, note)
            if key in active:
                st, vel = active.pop(key)
                if abs_tick > st:
                    spans.append(NoteSpan(st, abs_tick, note, vel, ch))

    spans.sort(key=lambda s: (s.start_tick, s.note))
    return spans


def _percentile(sorted_vals: List[int], p: float) -> int:
    """p in [0,1]"""
    if not sorted_vals:
        return -999
    idx = int(round((len(sorted_vals) - 1) * p))
    idx = max(0, min(idx, len(sorted_vals) - 1))
    return sorted_vals[idx]


def pick_rh_lh_tracks(mid: mido.MidiFile) -> Tuple[int, int]:
    """
    1) 若有 track_name 包含 right/left：直接用
    2) 否則：用「音高較高」的 track 當 RH（主旋律）
       - 用 p95（95百分位音高）當主判準，避免偶發超高音誤判
       - LH = 第二高
    """
    names = [(i, (get_track_name(tr) or "")) for i, tr in enumerate(mid.tracks)]
    rh = next((i for i, n in names if "right" in n.lower()), None)
    lh = next((i for i, n in names if "left" in n.lower()), None)
    if rh is not None and lh is not None and rh != lh:
        return rh, lh

    scored = []
    for i, tr in enumerate(mid.tracks):
        spans = extract_note_spans(tr)
        if not spans:
            continue

        # 避免鼓 track：channel 9(第10軌)常是鼓（幾乎全是 ch=9 就跳過）
        channels = [s.channel for s in spans]
        if channels and (sum(1 for c in channels if c == 9) / len(channels)) > 0.9:
            continue

        notes = sorted(s.note for s in spans)
        p95 = _percentile(notes, 0.95)
        mx = notes[-1]
        p50 = _percentile(notes, 0.50)
        scored.append(((p95, mx, p50), i))

    if len(scored) < 2:
        raise RuntimeError("找不到足夠的含音符 tracks 來分 RH/LH（請確認 MIDI 是否為鋼琴/多聲部）")

    scored.sort(key=lambda x: x[0])
    rh_i = scored[-1][1]
    lh_i = scored[-2][1]
    if rh_i == lh_i:
        raise RuntimeError("fallback 無法分出不同 RH/LH track")
    return rh_i, lh_i


# =========================
# 音域策略
# =========================
def apply_global_octave_shift(note: int, global_octave: int) -> int:
    """全體(左右手)升降八度，-1=降八度"""
    return note + global_octave * 12


def fit_right_hand_keep_all(note: int, play_min: int) -> int:
    """
    RH：全部保留（含和弦）
      - 超出上限 -> 往下折八度直到 <= PLAY_MAX
      - 超出下限 -> 往上折八度直到 >= play_min
      - 若原本很高(>=C5)，避免折到太低：至少留在中音(C4=60)以上（盡量）
    """
    orig = note
    n = note

    while n > PLAY_MAX:
        n -= 12
    while n < play_min:
        n += 12

    if orig >= HIGH_MIN and n < MID_MIN:
        n += 12
        if n > PLAY_MAX:
            n -= 12

    return n


def fit_left_hand_drop_too_low(note: int, play_min: int) -> Optional[int]:
    """
    LH：低於 play_min 直接忽略
    其餘若超高則往下折回範圍（較少見）
    """
    if note < play_min:
        return None
    n = note
    while n > PLAY_MAX:
        n -= 12
    while n < play_min:
        n += 12
    return n


# =========================
# MIDI note -> 遊戲按鍵（含 Shift/Ctrl + 高中低音列）
# =========================
def midi_to_game_token(note: int) -> str:
    core = PC_TO_TOKEN[note % 12]
    if note <= LOW_MAX:
        return "_" + core
    if note >= HIGH_MIN:
        return "^" + core
    return core


def token_to_keystroke(token: str) -> Tuple[Tuple[Key, ...], str, str]:
    """
    回傳：(mods_tuple, key_char, debug_str)
    mods_tuple 可能是 () / (Key.shift,) / (Key.ctrl,)
    """
    tok = token.strip()
    octave = "mid"
    if tok.startswith("^"):
        octave = "high"
        tok = tok[1:]
    elif tok.startswith("_"):
        octave = "low"
        tok = tok[1:]

    mods: List[Key] = []
    if tok in ACC_SHARP:
        mods.append(Key.shift)
        base = tok[1:]  # '#1' -> '1'
    elif tok in ACC_FLAT:
        mods.append(Key.ctrl)
        base = tok[1:]  # 'b3' -> '3'
    else:
        base = tok

    if base not in {"1", "2", "3", "4", "5", "6", "7"}:
        raise ValueError(f"Bad token core: {tok}")

    if octave == "low":
        key_char = LOW_KEYS[base]
        oct_mark = "_"
    elif octave == "high":
        key_char = HIGH_KEYS[base]
        oct_mark = "^"
    else:
        key_char = MID_KEYS[base]
        oct_mark = ""

    mod_txt = "+".join("SHIFT" if m == Key.shift else "CTRL" for m in mods)
    if mod_txt:
        mod_txt += "+"
    dbg = f"{mod_txt}{key_char} <= {oct_mark}{tok}"
    return tuple(mods), key_char, dbg


def press_group(mods: Tuple[Key, ...], keys: List[str], hold_s: float, debug: bool):
    """
    同 modifier 的 chord：先按 mods -> press keys -> hold -> release keys -> release mods
    """
    if debug:
        mtxt = "+".join("SHIFT" if m == Key.shift else "CTRL" for m in mods) or "NONE"
        print(f"    GROUP mods={mtxt} keys={keys} hold={hold_s:.3f}s")

    for m in mods:
        kbd.press(m)
    time.sleep(0.001)

    for k in keys:
        kbd.press(k)

    time.sleep(hold_s)

    for k in keys:
        kbd.release(k)

    time.sleep(0.001)
    for m in reversed(mods):
        kbd.release(m)


# =========================
# spans -> scheduled（秒）
# =========================
def spans_to_scheduled(
    spans: List[NoteSpan],
    hand: str,
    mid: mido.MidiFile,
    tempo_map: List[TempoChange],
    global_octave: int,
    play_min: int,
) -> List[ScheduledNote]:
    tpq = mid.ticks_per_beat
    out: List[ScheduledNote] = []

    for s in spans:
        n0 = apply_global_octave_shift(s.note, global_octave)

        if hand == "RH":
            n1 = fit_right_hand_keep_all(n0, play_min)
        else:
            n1_opt = fit_left_hand_drop_too_low(n0, play_min)
            if n1_opt is None:
                continue
            n1 = n1_opt

        t0 = tick_to_seconds(s.start_tick, tpq, tempo_map)
        t1 = tick_to_seconds(s.end_tick, tpq, tempo_map)
        dur = max(0.03, t1 - t0)

        out.append(ScheduledNote(t_sec=t0, dur_sec=dur, note=n1, hand=hand))

    out.sort(key=lambda x: (x.t_sec, x.note))
    return out


def group_chords(events: List[ScheduledNote], onset_eps: float) -> List[List[ScheduledNote]]:
    """同一時間點附近（<= onset_eps 秒）的事件視為同一 chord onset"""
    if not events:
        return []
    groups: List[List[ScheduledNote]] = []
    cur = [events[0]]
    for e in events[1:]:
        if abs(e.t_sec - cur[0].t_sec) <= onset_eps:
            cur.append(e)
        else:
            groups.append(cur)
            cur = [e]
    groups.append(cur)
    return groups


# =========================
# 解法1：同一拍只保留一種 modifier 群
# =========================
def choose_one_modifier_bucket(
    buckets: Dict[Tuple[Key, ...], List[str]],
    pick: str,
) -> Tuple[Tuple[Key, ...], List[str]]:
    """
    buckets: mods_tuple -> keys(list)
    pick:
      - most         : 保留音數最多的那群
      - prefer-none  : 有 NONE 就選 NONE，否則回 most
      - prefer-shift : 有 SHIFT 就選 SHIFT，否則回 most
      - prefer-ctrl  : 有 CTRL 就選 CTRL，否則回 most
    """
    uniq = {mods: sorted(set(keys)) for mods, keys in buckets.items()}
    if len(uniq) == 1:
        mods = next(iter(uniq.keys()))
        return mods, uniq[mods]

    def count(mods: Tuple[Key, ...]) -> int:
        return len(uniq[mods])

    def has_shift(mods: Tuple[Key, ...]) -> bool:
        return Key.shift in mods

    def has_ctrl(mods: Tuple[Key, ...]) -> bool:
        return Key.ctrl in mods

    if pick == "prefer-none":
        if () in uniq:
            return (), uniq[()]
    elif pick == "prefer-shift":
        cand = [m for m in uniq.keys() if has_shift(m)]
        if cand:
            cand.sort(key=count, reverse=True)
            m = cand[0]
            return m, uniq[m]
    elif pick == "prefer-ctrl":
        cand = [m for m in uniq.keys() if has_ctrl(m)]
        if cand:
            cand.sort(key=count, reverse=True)
            m = cand[0]
            return m, uniq[m]

    cand = list(uniq.keys())
    cand.sort(key=count, reverse=True)
    m = cand[0]
    return m, uniq[m]


# =========================
# 解法3：琶音刷和弦（極短、接近同時）
# =========================
def order_modifier_groups(
    uniq: Dict[Tuple[Key, ...], List[str]],
    first_pick: str,
) -> List[Tuple[Tuple[Key, ...], List[str]]]:
    """
    讓刷和弦順序穩定一點：
      - 若 prefer-* 命中，該群放最前
      - 其餘依音數多到少
    """
    items = [(mods, keys) for mods, keys in uniq.items()]

    def size(item):
        return len(item[1])

    preferred: Optional[Tuple[Key, ...]] = None
    if first_pick == "prefer-none":
        preferred = () if () in uniq else None
    elif first_pick == "prefer-shift":
        preferred = next((m for m in uniq.keys() if Key.shift in m), None)
    elif first_pick == "prefer-ctrl":
        preferred = next((m for m in uniq.keys() if Key.ctrl in m), None)

    if preferred is not None:
        first = [(preferred, uniq[preferred])]
        rest = [(m, k) for (m, k) in items if m != preferred]
        rest.sort(key=size, reverse=True)
        return first + rest

    items.sort(key=size, reverse=True)
    return items


# =========================
# 主播放（重點：絕對排程，不漂移）
# =========================
def play(
    mid_path: str,
    global_octave: int,
    left_drop_below: int,
    onset_eps: float,
    hold_ratio: float,
    start_delay: int,
    chord_mode: str,    # onegroup / arpeggio / raw
    chord_pick: str,    # most / prefer-*
    arp_gap_ms: float,
    arp_hold_ms: float,
    arp_max_groups: int,
    debug: bool,
    dry_run: bool,
):
    mid = mido.MidiFile(mid_path)
    tempo_map = build_tempo_map(mid)

    rh_i, lh_i = pick_rh_lh_tracks(mid)
    rh_name = get_track_name(mid.tracks[rh_i]) or f"Track{rh_i}"
    lh_name = get_track_name(mid.tracks[lh_i]) or f"Track{lh_i}"

    play_min = left_drop_below

    print("=== MIDI Piano Game Player ===")
    print(f"file={mid_path}")
    print(f"ticks_per_beat={mid.ticks_per_beat}, tracks={len(mid.tracks)}")
    print(f"RH: track {rh_i} name={rh_name!r}")
    print(f"LH: track {lh_i} name={lh_name!r}")
    print(f"global_octave_shift={global_octave} (octaves) => {global_octave * 12} semitones")
    print(f"left_drop_below/play_min={play_min} (MIDI note)")
    print(f"tempo_changes={len(tempo_map)}")
    if debug:
        print("Tempo map (first 20):")
        for c in tempo_map[:20]:
            bpm = mido.tempo2bpm(c.tempo_us_per_beat)
            print(f"  tick={c.abs_tick:>9} tempo={c.tempo_us_per_beat} us/beat bpm={bpm:.2f}")
    print(f"chord_mode={chord_mode}, chord_pick={chord_pick}")
    print(f"arp_gap_ms={arp_gap_ms}, arp_hold_ms={arp_hold_ms}, arp_max_groups={arp_max_groups}")
    print(f"dry_run={dry_run}, debug={debug}")
    print()

    rh_spans = extract_note_spans(mid.tracks[rh_i])
    lh_spans = extract_note_spans(mid.tracks[lh_i])

    rh_sched = spans_to_scheduled(rh_spans, "RH", mid, tempo_map, global_octave, play_min)
    lh_sched = spans_to_scheduled(lh_spans, "LH", mid, tempo_map, global_octave, play_min)

    all_sched = sorted(rh_sched + lh_sched, key=lambda x: (x.t_sec, x.note))
    chord_groups = group_chords(all_sched, onset_eps=onset_eps)

    if not chord_groups:
        print("No playable events after filtering.")
        return

    for i in range(start_delay, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)
    print("PLAY!\n")

    # 絕對排程：不累積漂移（變速點會準）
    song_start = time.perf_counter()

    gap_s = max(0.0, arp_gap_ms / 1000.0)
    # 每群按住多久：專門給 arpeggio 用（人耳分不出通常 8~12ms）
    arp_hold_s = max(0.004, arp_hold_ms / 1000.0)  # 最少 4ms，避免太快遊戲吃不到

    for gi, group in enumerate(chord_groups):
        onset = group[0].t_sec
        target = song_start + onset

        while True:
            now = time.perf_counter()
            remain = target - now
            if remain <= 0:
                break
            time.sleep(min(remain, 0.01))

        # 一般（非 arpeggio）按住時間
        min_dur = min(e.dur_sec for e in group)
        hold = max(0.02, min_dur * hold_ratio)

        buckets: Dict[Tuple[Key, ...], List[str]] = defaultdict(list)
        dbg_items: List[str] = []

        for e in group:
            token = midi_to_game_token(e.note)
            mods, key_char, dbg = token_to_keystroke(token)
            buckets[mods].append(key_char)
            if debug:
                dbg_items.append(f"{e.hand}:{e.note}->{token} [{dbg}]")

        uniq = {mods: sorted(set(keys)) for mods, keys in buckets.items()}

        if debug:
            now_s = time.perf_counter() - song_start
            print(f"[{gi:05d}] now={now_s:8.3f}s onset={onset:8.3f}s notes={len(group)} groups={len(uniq)}")
            for s in dbg_items[:25]:
                print("   ", s)
            if len(dbg_items) > 25:
                print("   ...")

        if dry_run:
            continue

        if chord_mode == "raw":
            # 直接每群都送（常會怪）
            for mods, keys in uniq.items():
                press_group(mods, keys, hold_s=hold, debug=debug)

        elif chord_mode == "onegroup":
            # 解法1：只保留一群 modifier
            chosen_mods, chosen_keys = choose_one_modifier_bucket(uniq, chord_pick)
            if debug:
                print(f"   -> ONEGROUP keep mods={chosen_mods} keys={chosen_keys} hold={hold:.3f}s")
            press_group(chosen_mods, chosen_keys, hold_s=hold, debug=debug)

        elif chord_mode == "arpeggio":
            # 解法3：把不同 modifier 群「極短」刷出來（接近同時）
            ordered = order_modifier_groups(uniq, chord_pick)[:max(1, arp_max_groups)]
            if debug:
                print(f"   -> ARPEGGIO groups={len(ordered)} gap={gap_s:.3f}s hold={arp_hold_s:.3f}s")

            for idx, (mods, keys) in enumerate(ordered):
                press_group(mods, keys, hold_s=arp_hold_s, debug=debug)
                if idx != len(ordered) - 1 and gap_s > 0:
                    time.sleep(gap_s)

        else:
            raise ValueError(f"Unknown chord_mode: {chord_mode}")

    print("\nDONE.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mid", help="input .mid file path")
    ap.add_argument("--global-octave", type=int, default=0,
                    help="全體(左右手)升降八度：-1=整首降八度（你自己選）")
    ap.add_argument("--left-drop-below", type=int, default=48,
                    help="左手低於此 MIDI note 直接忽略（同時也當作 play_min，下限）")
    ap.add_argument("--onset-eps", type=float, default=0.020,
                    help="同一和弦 onset 判定窗（秒）")
    ap.add_argument("--hold-ratio", type=float, default=0.35,
                    help="（非 arpeggio）按住時間 = chord最短dur * hold_ratio")
    ap.add_argument("--start-delay", type=int, default=4)

    ap.add_argument("--chord-mode", choices=["onegroup", "arpeggio", "raw"], default="arpeggio",
                    help="onegroup=解法1(只留一群modifier) / arpeggio=解法3(刷和弦) / raw=不處理")
    ap.add_argument("--chord-pick", choices=["most", "prefer-none", "prefer-shift", "prefer-ctrl"], default="most",
                    help="onegroup：混合modifier時選哪群；arpeggio：哪群優先刷出")
    ap.add_argument("--arp-gap-ms", type=float, default=0.0,
                    help="arpeggio模式：群與群之間的間隔(ms)，要快就 0~2")
    ap.add_argument("--arp-hold-ms", type=float, default=8.0,
                    help="arpeggio模式：每一群按住時間(ms)，要快就 8~12")
    ap.add_argument("--arp-max-groups", type=int, default=3,
                    help="arpeggio模式：同一拍最多刷幾群（2~3 常用）")

    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="只印不送鍵（驗證 tempo/分手/半音/八度/和弦策略）")
    args = ap.parse_args()

    play(
        mid_path=args.mid,
        global_octave=args.global_octave,
        left_drop_below=args.left_drop_below,
        onset_eps=args.onset_eps,
        hold_ratio=args.hold_ratio,
        start_delay=args.start_delay,
        chord_mode=args.chord_mode,
        chord_pick=args.chord_pick,
        arp_gap_ms=args.arp_gap_ms,
        arp_hold_ms=args.arp_hold_ms,
        arp_max_groups=args.arp_max_groups,
        debug=args.debug,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
