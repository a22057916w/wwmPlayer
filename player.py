import time
import random
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Set

import mido
from pynput.keyboard import Controller, Key

kbd = Controller()

# ----------------------------
# Your game mapping
# ----------------------------
MID  = {"1":"a","2":"s","3":"d","4":"f","5":"g","6":"h","7":"j"}  # middle
HIGH = {"1":"q","2":"w","3":"e","4":"r","5":"t","6":"y","7":"u"}  # +1 octave
LOW  = {"1":"z","2":"x","3":"c","4":"v","5":"b","6":"n","7":"m"}  # -1 octave

ACC_SHARP = {"#1", "#4", "#5"}  # Shift + [1/4/5]
ACC_FLAT  = {"b3", "b7"}        # Ctrl  + [3/7]

# "1=C" chromatic mapping (limited to your game's supported accidentals)
PC_TO_TOKEN = {
    0: "1",    # C
    1: "#1",   # C#
    2: "2",    # D
    3: "b3",   # Eb
    4: "3",    # E
    5: "4",    # F
    6: "#4",   # F#
    7: "5",    # G
    8: "#5",   # G#
    9: "6",    # A
    10:"b7",   # Bb
    11:"7",    # B
}

# Game playable 3 octaves buckets:
LOW_MIN, HIGH_MAX = 48, 83   # C3..B5 (Z..Q rows)
MID_MIN, MID_MAX  = 60, 71   # C4..B4 (A row)  <-- target center range
MID_CENTER = 66              # around F#4/Gb4, just a numeric center target

@dataclass
class Span:
    start: float
    end: float
    midi: int
    vel: int
    ch: int

@dataclass
class MelNote:
    t: float       # onset time (seconds)
    midi: int      # original midi note
    dur: float     # duration until next onset (seconds)

@dataclass
class Event:
    token: str
    seconds: float
    orig_midi: int
    shifted_midi: int

# ----------------------------
# MIDI parsing (track or merged)
# ----------------------------
def parse_spans_merged(mid: mido.MidiFile, ignore_drums: bool = True) -> List[Span]:
    tempo = 500000
    t = 0.0
    active: Dict[Tuple[int,int], Tuple[float,int]] = {}
    spans: List[Span] = []

    for msg in mido.merge_tracks(mid.tracks):
        if msg.time:
            t += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)

        if msg.type == "set_tempo":
            tempo = msg.tempo

        if msg.type in ("note_on","note_off"):
            ch = getattr(msg, "channel", 0)
            if ignore_drums and ch == 9:
                continue

            note = msg.note
            if msg.type == "note_on" and msg.velocity > 0:
                active[(ch, note)] = (t, msg.velocity)
            else:
                key = (ch, note)
                if key in active:
                    st, v = active.pop(key)
                    if t > st:
                        spans.append(Span(st, t, note, v, ch))

    end_t = t
    for (ch, note), (st, v) in active.items():
        spans.append(Span(st, end_t + 0.12, note, v, ch))

    spans.sort(key=lambda s: s.start)
    return spans

def parse_spans_track(mid: mido.MidiFile, track_index: int, ignore_drums: bool = True) -> List[Span]:
    tempo = 500000
    t = 0.0
    active: Dict[Tuple[int,int], Tuple[float,int]] = {}
    spans: List[Span] = []

    tr = mid.tracks[track_index]
    for msg in tr:
        if msg.time:
            t += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)

        if msg.type == "set_tempo":
            tempo = msg.tempo

        if msg.type in ("note_on","note_off"):
            ch = getattr(msg, "channel", 0)
            if ignore_drums and ch == 9:
                continue

            note = msg.note
            if msg.type == "note_on" and msg.velocity > 0:
                active[(ch, note)] = (t, msg.velocity)
            else:
                key = (ch, note)
                if key in active:
                    st, v = active.pop(key)
                    if t > st:
                        spans.append(Span(st, t, note, v, ch))

    end_t = t
    for (ch, note), (st, v) in active.items():
        spans.append(Span(st, end_t + 0.12, note, v, ch))

    spans.sort(key=lambda s: s.start)
    return spans

# ----------------------------
# Melody extraction: group chord onsets -> pick highest note per onset
# ----------------------------
def spans_to_melody(spans: List[Span], chord_eps: float) -> List[MelNote]:
    if not spans:
        return []

    groups: List[List[Span]] = []
    cur = [spans[0]]
    for s in spans[1:]:
        if abs(s.start - cur[0].start) <= chord_eps:
            cur.append(s)
        else:
            groups.append(cur)
            cur = [s]
    groups.append(cur)

    # pick melody note = highest pitch in group
    onsets: List[Tuple[float, int]] = []
    for g in groups:
        t0 = g[0].start
        mel = max(g, key=lambda x: x.midi).midi
        onsets.append((t0, mel))

    # durations based on next onset time
    melody: List[MelNote] = []
    for i, (t0, n) in enumerate(onsets):
        if i + 1 < len(onsets):
            dur = max(0.03, onsets[i+1][0] - t0)
        else:
            # last: just a small tail
            dur = 0.25
        melody.append(MelNote(t=t0, midi=n, dur=dur))
    return melody

# ----------------------------
# Phrase segmentation: split when time gap > phrase_gap
# ----------------------------
def split_phrases(melody: List[MelNote], phrase_gap: float) -> List[List[MelNote]]:
    if not melody:
        return []
    phrases: List[List[MelNote]] = []
    cur = [melody[0]]
    for prev, nxt in zip(melody, melody[1:]):
        gap = max(0.0, nxt.t - prev.t)
        if gap > phrase_gap:
            phrases.append(cur)
            cur = [nxt]
        else:
            cur.append(nxt)
    phrases.append(cur)
    return phrases

# ----------------------------
# Choose ONE octave shift (multiple of 12) per phrase to center into MID range
# ----------------------------
def phrase_cost(notes: List[int], shift: int) -> float:
    """
    Lower is better.
    Penalize notes outside game range hard.
    Prefer notes inside MID range (60..71) and close to MID_CENTER.
    """
    cost = 0.0
    for n in notes:
        nn = n + shift

        # out of game range -> huge penalty
        if nn < LOW_MIN or nn > HIGH_MAX:
            cost += 1e6 + (abs(nn - (LOW_MIN if nn < LOW_MIN else HIGH_MAX)) * 500.0)
            continue

        # inside mid range -> small cost by distance to center
        if MID_MIN <= nn <= MID_MAX:
            cost += (nn - MID_CENTER) ** 2
        else:
            # outside mid range but still playable -> penalty to encourage mid
            # slightly prefer near mid edges
            dist = (MID_MIN - nn) if nn < MID_MIN else (nn - MID_MAX)
            cost += 200.0 + (dist ** 2) * 10.0
    return cost

def choose_phrase_shift(phrase: List[MelNote], base_transpose: int = 0) -> int:
    notes = [m.midi + base_transpose for m in phrase]
    # try shifts by octaves
    candidates = [12*k for k in range(-6, 7)]
    best_shift = 0
    best_cost = float("inf")
    for sh in candidates:
        c = phrase_cost(notes, sh)
        if c < best_cost:
            best_cost = c
            best_shift = sh
    return best_shift

# ----------------------------
# MIDI note -> token (after transpose+shift)
# ----------------------------
def midi_to_token(n: int) -> str:
    pc = n % 12
    core = PC_TO_TOKEN[pc]
    if n < 60:
        return "_" + core
    elif n >= 72:
        return "^" + core
    else:
        return core

def token_to_stroke(token: str) -> Optional[Tuple[List[Key], str, str]]:
    token = token.strip()
    if token == "0":
        return None

    octave = "mid"
    if token.startswith("^"):
        octave = "high"
        token = token[1:]
    elif token.startswith("_"):
        octave = "low"
        token = token[1:]

    mods: List[Key] = []
    degree = token  # "5" "#4" "b7"

    if degree in ACC_SHARP:
        mods.append(Key.shift)
        base_degree = degree[1:]
    elif degree in ACC_FLAT:
        mods.append(Key.ctrl)
        base_degree = degree[1:]
    else:
        base_degree = degree

    if base_degree not in {"1","2","3","4","5","6","7"}:
        raise ValueError(f"Bad token: {token}")

    if octave == "mid":
        key_char = MID[base_degree]
        oct_tag = ""
    elif octave == "high":
        key_char = HIGH[base_degree]
        oct_tag = "^"
    else:
        key_char = LOW[base_degree]
        oct_tag = "_"

    mod_txt = "+".join(["SHIFT" if m == Key.shift else "CTRL" for m in mods])
    if mod_txt:
        mod_txt += "+"
    dbg = f"{mod_txt}{key_char}({oct_tag}{degree})"
    return mods, key_char, dbg

def press_stroke(stroke: Tuple[List[Key], str, str], hold_s: float, dry_run: bool):
    mods, key_char, _ = stroke
    mods_present: Set[Key] = set(mods)

    # stable order
    mods_unique = []
    for m in [Key.shift, Key.ctrl]:
        if m in mods_present:
            mods_unique.append(m)

    # print physical sequence
    seq = []
    for m in mods_unique:
        seq.append("SHIFT↓" if m == Key.shift else "CTRL↓")
    seq.append(f"{key_char}↓")
    seq.append(f"{key_char}↑")
    for m in reversed(mods_unique):
        seq.append("CTRL↑" if m == Key.ctrl else "SHIFT↑")
    print("    SEND:", " ".join(seq))

    if dry_run:
        time.sleep(hold_s)
        return

    for m in mods_unique:
        kbd.press(m)
    time.sleep(0.01)

    kbd.press(key_char)
    time.sleep(hold_s)
    kbd.release(key_char)

    time.sleep(0.005)
    for m in reversed(mods_unique):
        kbd.release(m)

# ----------------------------
# Build final playable events
# ----------------------------
def build_events_phrase_center(melody: List[MelNote], phrase_gap: float, base_transpose: int) -> List[Event]:
    phrases = split_phrases(melody, phrase_gap=phrase_gap)
    events: List[Event] = []

    for p_idx, ph in enumerate(phrases):
        phrase_shift = choose_phrase_shift(ph, base_transpose=base_transpose)
        # apply this shift to whole phrase
        for m in ph:
            shifted = m.midi + base_transpose + phrase_shift

            # hard clamp into game range by octave if needed (rare after scoring)
            while shifted < LOW_MIN:
                shifted += 12
            while shifted > HIGH_MAX:
                shifted -= 12

            tok = midi_to_token(shifted)
            events.append(Event(token=tok, seconds=m.dur, orig_midi=m.midi, shifted_midi=shifted))

    return events

def play(events: List[Event], start_delay: int, hold_ratio: float, jitter_ms: int, dry_run: bool):
    print("=== Phrase Octave-Center AutoPlayer (melody-only) ===")
    print("建議：英文輸入法 + 遊戲視窗置前 + 以管理員執行 Python")
    for i in range(start_delay, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)
    print("PLAY!\n")

    logical_time = 0.0
    for idx, ev in enumerate(events):
        dur = ev.seconds
        hold = max(0.02, dur * hold_ratio)
        jitter = random.uniform(-jitter_ms, jitter_ms) / 1000.0

        stroke = token_to_stroke(ev.token)
        if stroke is None:
            time.sleep(dur)
            logical_time += dur
            continue

        _, _, dbg = stroke
        print(f"[{idx:04d}] +{logical_time:7.3f}s  MIDI {ev.orig_midi:3d} -> {ev.shifted_midi:3d} -> {ev.token:>4s} -> {dbg} | hold={hold:.3f}s dur={dur:.3f}s")
        press_stroke(stroke, hold_s=hold, dry_run=dry_run)

        time.sleep(max(0.0, dur - hold) + jitter)
        logical_time += dur

    print("\nDONE.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mid", help="path to .mid file")
    ap.add_argument("--merge-tracks", action="store_true", help="merge all tracks into one timeline (default)")
    ap.add_argument("--track", type=int, default=0, help="use a specific track if not merging")
    ap.add_argument("--chord-eps", type=float, default=0.02, help="seconds: group notes starting together")
    ap.add_argument("--phrase-gap", type=float, default=0.60, help="seconds: split phrase when onset gap > this")
    ap.add_argument("--transpose", type=int, default=0, help="semitones (+/-) before phrase octave shifting")
    ap.add_argument("--start-delay", type=int, default=4)
    ap.add_argument("--hold", type=float, default=0.55)
    ap.add_argument("--jitter-ms", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-ignore-drums", action="store_true")
    args = ap.parse_args()

    mid = mido.MidiFile(args.mid)
    ignore_drums = not args.no_ignore_drums

    if args.merge_tracks:
        spans = parse_spans_merged(mid, ignore_drums=ignore_drums)
        print(f"tracks={len(mid.tracks)} | spans(merged)={len(spans)}")
    else:
        if not (0 <= args.track < len(mid.tracks)):
            raise SystemExit(f"--track out of range. tracks={len(mid.tracks)}")
        spans = parse_spans_track(mid, args.track, ignore_drums=ignore_drums)
        print(f"tracks={len(mid.tracks)} | spans(track {args.track})={len(spans)}")

    melody = spans_to_melody(spans, chord_eps=args.chord_eps)
    print(f"melody_onsets={len(melody)}")

    events = build_events_phrase_center(melody, phrase_gap=args.phrase_gap, base_transpose=args.transpose)
    print(f"events={len(events)}\n")

    play(events, start_delay=args.start_delay, hold_ratio=args.hold, jitter_ms=args.jitter_ms, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
