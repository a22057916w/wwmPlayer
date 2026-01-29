"""
midi_dump_to_txt.py

目的：
  讀取 .mid 並把「全部事件」輸出到一個 .txt 檔（含：
    - tempo / 拍號 / 調號（樂理基礎）
    - note_on / note_off（音符、音高、力度、通道）
    - program_change（樂器）
    - control_change（踏板/音量等）
  並同時把 tick -> 秒（sec）算出來（tempo map 分段換算，避免節奏解析錯）。

安裝：
  pip install mido

使用：
  python midi_dump_to_txt.py FlowerDance.mid FlowerDance_dump.txt
  python midi_dump_to_txt.py FlowerDance.mid FlowerDance_dump.txt --only-meta
  python midi_dump_to_txt.py FlowerDance.mid FlowerDance_dump.txt --max-events 20000
  python midi_dump_to_txt.py FlowerDance.mid FlowerDance_dump.txt --track 0
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import mido


# ---------------------------
# 樂理：MIDI 音高 -> 音名（十二平均律）
# ---------------------------
NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def midi_note_to_name(midi_note: int) -> str:
    """
    MIDI note number (0~127) -> note name (e.g. 60 -> C4)
    標準：C4 = 60
    octave = (note // 12) - 1
    """
    pc = midi_note % 12
    octave = (midi_note // 12) - 1
    return f"{NOTE_NAMES_SHARP[pc]}{octave}"


# ---------------------------
# 樂理：Tempo / 拍號 / 調號
# ---------------------------
@dataclass
class TempoChange:
    abs_tick: int
    tempo_us_per_beat: int  # 1 beat(通常四分音符) 的微秒數
    bpm: float              # BPM: 每分鐘幾拍（拍 = 拍號裡定義的 beat，通常以四分音符作基準）

@dataclass
class TimeSigChange:
    abs_tick: int
    numerator: int
    denominator: int

@dataclass
class KeySigChange:
    abs_tick: int
    key: str  # 例如 'C', 'G', 'F#', 'Bb', 'a', 'c#'


def build_tempo_map(mid: mido.MidiFile) -> List[TempoChange]:
    """
    MIDI tempo 用 meta message: set_tempo 表示。
    - 有些檔案 tempo 全放在 Track 0
    - 也可能中途多次變速（rit./accel.）=> set_tempo 會出現很多次
    """
    changes: List[TempoChange] = []

    # MIDI 預設 tempo：500000 us/beat = 120 BPM
    default_tempo = 500000
    changes.append(TempoChange(0, default_tempo, mido.tempo2bpm(default_tempo)))

    for tr in mid.tracks:
        abs_tick = 0
        for msg in tr:
            abs_tick += msg.time
            if msg.type == "set_tempo":
                changes.append(TempoChange(abs_tick, msg.tempo, mido.tempo2bpm(msg.tempo)))

    changes.sort(key=lambda x: x.abs_tick)
    return changes


def build_timesig_map(mid: mido.MidiFile) -> List[TimeSigChange]:
    """
    time_signature：拍號（例如 4/4、3/4、6/8）
    樂理意義：
      4/4 => 每小節 4 拍、以四分音符當 1 拍
      6/8 => 每小節 6 個八分音符（常見 feel 是 2 拍重音）
    """
    changes: List[TimeSigChange] = []
    for tr in mid.tracks:
        abs_tick = 0
        for msg in tr:
            abs_tick += msg.time
            if msg.type == "time_signature":
                changes.append(TimeSigChange(abs_tick, msg.numerator, msg.denominator))
    changes.sort(key=lambda x: x.abs_tick)
    return changes


def build_keysig_map(mid: mido.MidiFile) -> List[KeySigChange]:
    """
    key_signature：調號（例如 B 大調 / c# 小調）
    樂理意義：
      - 決定主音(Do)是哪個音
      - 哪些音預設要升(#)或降(b)
    """
    changes: List[KeySigChange] = []
    for tr in mid.tracks:
        abs_tick = 0
        for msg in tr:
            abs_tick += msg.time
            if msg.type == "key_signature":
                changes.append(KeySigChange(abs_tick, msg.key))
    changes.sort(key=lambda x: x.abs_tick)
    return changes


def tick_to_seconds(abs_tick: int, ticks_per_beat: int, tempo_map: List[TempoChange]) -> float:
    """
    把「絕對 tick」換成「絕對秒數」：
      - 需要 tempo_map（可能中途變速）
      - 用分段積分的方式換算（避免節奏整首跑掉）

    公式：
      delta_beat = delta_tick / ticks_per_beat
      delta_sec  = delta_beat * (tempo_us_per_beat / 1e6)
    """
    total_sec = 0.0
    for i in range(len(tempo_map)):
        t0 = tempo_map[i].abs_tick
        tempo_us = tempo_map[i].tempo_us_per_beat
        t1 = tempo_map[i + 1].abs_tick if i + 1 < len(tempo_map) else abs_tick

        if abs_tick <= t0:
            break

        seg_end = min(abs_tick, t1)
        if seg_end > t0:
            delta_tick = seg_end - t0
            delta_beat = delta_tick / ticks_per_beat
            total_sec += delta_beat * (tempo_us / 1_000_000.0)

        if abs_tick < t1:
            break

    return total_sec


def tick_to_bar_beat(abs_tick: int, ticks_per_beat: int, timesig_map: List[TimeSigChange]) -> Tuple[int, float, Tuple[int, int]]:
    """
    估算 abs_tick 落在：
      - bar：第幾小節（從 1 開始）
      - beat：小節內第幾拍（可小數）
    用途：你 debug 節奏「有沒有落在拍點」很有用。

    注意：這裡是以「最後一個 <= abs_tick 的拍號」為基準做近似。
    """
    if not timesig_map:
        numer, denom = 4, 4
    else:
        numer, denom = timesig_map[0].numerator, timesig_map[0].denominator
        for ts in timesig_map:
            if ts.abs_tick <= abs_tick:
                numer, denom = ts.numerator, ts.denominator
            else:
                break

    # ticks_per_beat 通常以「四分音符」為基準
    tick_per_score_beat = ticks_per_beat * (4 / denom)
    bar_len = numer * tick_per_score_beat

    bar = int(abs_tick // bar_len) + 1
    beat = (abs_tick % bar_len) / tick_per_score_beat + 1
    return bar, beat, (numer, denom)


def dump_to_text(mid: mido.MidiFile, only_meta: bool, max_events: Optional[int], track_filter: Optional[int]) -> str:
    tempo_map = build_tempo_map(mid)
    timesig_map = build_timesig_map(mid)
    keysig_map = build_keysig_map(mid)

    lines: List[str] = []
    lines.append("========== MIDI HEADER ==========")
    lines.append(f"mid.type={mid.type}  (0=single-track, 1=multi-track, 2=async)")
    lines.append(f"ticks_per_beat(PPQ)={mid.ticks_per_beat}")
    lines.append(f"tracks={len(mid.tracks)}")
    lines.append("")

    lines.append("========== GLOBAL META (collected) ==========")
    lines.append(f"Tempo changes count={len(tempo_map)}")
    for tc in tempo_map[:200]:
        lines.append(f"  tick={tc.abs_tick:>9} tempo={tc.tempo_us_per_beat} us/beat  bpm={tc.bpm:.3f}")
    if len(tempo_map) > 200:
        lines.append("  ... (more tempo changes)")
    lines.append("")

    lines.append(f"Time signature changes count={len(timesig_map)}")
    for ts in timesig_map[:200]:
        lines.append(f"  tick={ts.abs_tick:>9}  {ts.numerator}/{ts.denominator}")
    if len(timesig_map) > 200:
        lines.append("  ... (more time signatures)")
    lines.append("")

    lines.append(f"Key signature changes count={len(keysig_map)}")
    for ks in keysig_map[:200]:
        lines.append(f"  tick={ks.abs_tick:>9}  key={ks.key}")
    if len(keysig_map) > 200:
        lines.append("  ... (more key signatures)")
    lines.append("============================================")
    lines.append("")

    for ti, tr in enumerate(mid.tracks):
        if track_filter is not None and ti != track_filter:
            continue

        lines.append("")
        lines.append("############################")
        lines.append(f"### TRACK {ti} DETAILED DUMP")
        lines.append("############################")

        abs_tick = 0
        shown = 0
        for msg in tr:
            abs_tick += msg.time

            if only_meta and not msg.is_meta:
                continue

            sec = tick_to_seconds(abs_tick, mid.ticks_per_beat, tempo_map)
            bar, beat, ts = tick_to_bar_beat(abs_tick, mid.ticks_per_beat, timesig_map)

            # 針對常見訊息加樂理/播放意義的補充欄位
            extra = ""
            if msg.type in ("note_on", "note_off"):
                extra = f" | note={msg.note}({midi_note_to_name(msg.note)}) ch={getattr(msg,'channel',None)} vel={getattr(msg,'velocity',None)}"
                # 樂理：note_on 表示音開始；note_off 表示音結束（或 note_on vel=0）
            elif msg.type == "set_tempo":
                extra = f" | tempo={msg.tempo} us/beat -> bpm={mido.tempo2bpm(msg.tempo):.3f}"
                # 樂理：tempo 改了 => tick->sec 會改 => 節奏會變
            elif msg.type == "time_signature":
                extra = f" | time_sig={msg.numerator}/{msg.denominator}"
                # 樂理：每小節幾拍、以哪種音符為一拍
            elif msg.type == "key_signature":
                extra = f" | key_sig={msg.key}"
                # 樂理：大調/小調的調號資訊（升降記號）
            elif msg.type == "program_change":
                extra = f" | program={msg.program} ch={getattr(msg,'channel',None)}"
                # 播放：換樂器音色
            elif msg.type == "control_change":
                extra = f" | ctrl={msg.control} value={msg.value} ch={getattr(msg,'channel',None)}"
                # 播放：踏板/音量/表情等

            lines.append(
                f"tick={abs_tick:>9} sec={sec:>9.3f} bar={bar:>4} beat={beat:>6.2f} "
                f"ts={ts[0]}/{ts[1]} type={msg.type:<14} msg={msg}{extra}"
            )

            shown += 1
            if max_events is not None and shown >= max_events:
                lines.append(f"--- STOP: reached --max-events {max_events} ---")
                break

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mid", help="input .mid file path")
    ap.add_argument("out", help="output .txt file path")
    ap.add_argument("--only-meta", action="store_true", help="only dump meta events (tempo/time/key/text)")
    ap.add_argument("--max-events", type=int, default=None, help="limit total printed events per track (None=all)")
    ap.add_argument("--track", type=int, default=None, help="dump only one track index (e.g. 0)")
    args = ap.parse_args()

    mid = mido.MidiFile(args.mid)
    text = dump_to_text(mid, only_meta=args.only_meta, max_events=args.max_events, track_filter=args.track)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
