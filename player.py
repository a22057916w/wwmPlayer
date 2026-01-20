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

# "1=C" mapping into the only accidentals your game supports
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

# Register ranges (MIDI notes) corresponding to your 3 rows:
LOW_MIN, LOW_MAX   = 48, 59   # C3..B3  (z..m)
MID_MIN, MID_MAX   = 60, 71   # C4..B4  (a..j)
HIGH_MIN, HIGH_MAX = 72, 83   # C5..B5  (q..u)

PLAY_MIN, PLAY_MAX = LOW_MIN, HIGH_MAX

@dataclass
class Span:
    start: float
    end: float
    midi: int
    vel: int
    ch: int

@dataclass
class MelNote:
    t: float
    midi: int
    dur: float

@dataclass
class Event:
    token: str
    seconds: float
    orig_midi: int
    shifted_midi: int
    r: float
    reg: str

# ----------------------------
# Key signature -> transpose-to-C
# ----------------------------
def _parse_key_signature_to_tonic_pc(key: str) -> int:
    """
    mido key examples:
      Major: 'C', 'G', 'F#', 'Bb'
      Minor: 'a', 'f#', 'eb' (lowercase)
    Return tonic pitch-class: C=0..B=11
    """
    key = key.strip()
    tonic_map = {
        "C":0, "C#":1, "Db":1, "D":2, "D#":3, "Eb":3, "E":4, "F":5,
        "F#":6, "Gb":6, "G":7, "G#":8, "Ab":8, "A":9, "A#":10, "Bb":10, "B":11,
    }
    if not key:
        raise ValueError("Empty key")

    if key[0].islower():  # minor
        k = key[0].upper() + key[1:]  # 'f#'->'F#'
        if k in tonic_map:
            return tonic_map[k]
        raise ValueError(f"Unsupported minor key format: {key}")

    if key in tonic_map:
        return tonic_map[key]
    raise ValueError(f"Unsupported key signature: {key}")

def detect_key_signature(mid: mido.MidiFile) -> Optional[str]:
    for tr in mid.tracks:
        for msg in tr:
            if msg.type == "key_signature":
                return msg.key
    return None

def transpose_to_c_from_key(key: str) -> int:
    """
    Compute semitone shift to move tonic -> C.
    Return small shift in [-6..+6] when possible.
    """
    tonic = _parse_key_signature_to_tonic_pc(key)  # 0..11
    t = (-tonic) % 12  # 0..11
    if t > 6:
        t -= 12
    return t

# ----------------------------
# MIDI parsing
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

        if msg.type in ("note_on", "note_off"):
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

        if msg.type in ("note_on", "note_off"):
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
# Melody extraction (highest note per onset)
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

    onsets: List[Tuple[float, int]] = []
    for g in groups:
        t0 = g[0].start
        mel = max(g, key=lambda x: x.midi).midi
        onsets.append((t0, mel))

    melody: List[MelNote] = []
    for i, (t0, n) in enumerate(onsets):
        if i + 1 < len(onsets):
            dur = max(0.03, onsets[i+1][0] - t0)
        else:
            dur = 0.25
        melody.append(MelNote(t=t0, midi=n, dur=dur))
    return melody

# ----------------------------
# Bucket normalize (your "0..100 -> 0..1 -> 3 registers") + octave-fit
# ----------------------------
def _relative_height(note: int, nmin: int, nmax: int) -> float:
    if nmax <= nmin:
        return 0.5
    return (note - nmin) / (nmax - nmin)

def _choose_register(r: float) -> str:
    if r < (1.0/3.0):
        return "low"
    elif r < (2.0/3.0):
        return "mid"
    else:
        return "high"

def _fit_to_register(note: int, reg: str) -> int:
    if reg == "low":
        rmin, rmax = LOW_MIN, LOW_MAX
    elif reg == "mid":
        rmin, rmax = MID_MIN, MID_MAX
    else:
        rmin, rmax = HIGH_MIN, HIGH_MAX

    cand = note
    while cand > rmax:
        cand -= 12
    while cand < rmin:
        cand += 12

    while cand < PLAY_MIN:
        cand += 12
    while cand > PLAY_MAX:
        cand -= 12

    return cand

def midi_to_token(n: int) -> str:
    pc = n % 12
    core = PC_TO_TOKEN[pc]
    if n <= LOW_MAX:
        return "_" + core
    elif n >= HIGH_MIN:
        return "^" + core
    else:
        return core

# ----------------------------
# Token -> key stroke
# ----------------------------
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
    degree = token  # "5", "#4", "b7"

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

    mods_unique: List[Key] = []
    for m in [Key.shift, Key.ctrl]:
        if m in mods_present:
            mods_unique.append(m)

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
# Build events
# ----------------------------
def build_events(melody: List[MelNote], total_transpose: int) -> List[Event]:
    if not melody:
        return []

    notes_t = [m.midi + total_transpose for m in melody]
    nmin = min(notes_t)
    nmax = max(notes_t)

    events: List[Event] = []
    for m in melody:
        orig = m.midi
        n = orig + total_transpose

        r = _relative_height(n, nmin, nmax)
        reg = _choose_register(r)

        shifted = _fit_to_register(n, reg)
        token = midi_to_token(shifted)

        events.append(Event(
            token=token,
            seconds=m.dur,
            orig_midi=orig,
            shifted_midi=shifted,
            r=r,
            reg=reg
        ))
    return events

def play(events: List[Event], start_delay: int, hold_ratio: float, jitter_ms: int, dry_run: bool):
    print("=== Range-Bucket + KeyFix AutoPlayer (melody-only) ===")
    print("Logic: (1) transpose-to-C (optional) -> (2) relative height bucket -> (3) octave-fit into row")
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
        print(f"[{idx:04d}] +{logical_time:7.3f}s  MIDI {ev.orig_midi:3d} -> {ev.shifted_midi:3d}"
              f" | r={ev.r:5.2f} reg={ev.reg:>4s} | {ev.token:>4s} -> {dbg}"
              f" | hold={hold:.3f}s dur={dur:.3f}s")

        press_stroke(stroke, hold_s=hold, dry_run=dry_run)

        time.sleep(max(0.0, dur - hold) + jitter)
        logical_time += dur

    print("\nDONE.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mid", help="path to .mid file")
    ap.add_argument("--merge-tracks", action="store_true")
    ap.add_argument("--track", type=int, default=0)
    ap.add_argument("--chord-eps", type=float, default=0.02)

    ap.add_argument("--normalize-to-c", action="store_true",
                    help="use MIDI key_signature (if present) to transpose tonic->C before mapping")
    ap.add_argument("--key", type=str, default=None,
                    help="manual key signature, e.g. C, G, F#, Bb, a, f#, eb (used if --normalize-to-c has no key_signature)")
    ap.add_argument("--transpose", type=int, default=0,
                    help="additional manual semitone transpose (+/-)")

    ap.add_argument("--start-delay", type=int, default=4)
    ap.add_argument("--hold", type=float, default=0.55)
    ap.add_argument("--jitter-ms", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-ignore-drums", action="store_true")
    args = ap.parse_args()

    mid = mido.MidiFile(args.mid)
    ignore_drums = not args.no_ignore_drums

    auto_t = 0
    if args.normalize_to_c:
        ks = detect_key_signature(mid)
        if ks is not None:
            auto_t = transpose_to_c_from_key(ks)
            print(f"Detected key_signature={ks}, auto transpose-to-C = {auto_t} semitones")
        elif args.key:
            auto_t = transpose_to_c_from_key(args.key)
            print(f"No key_signature in MIDI, using --key={args.key}, transpose-to-C = {auto_t} semitones")
        else:
            print("No key_signature in MIDI and no --key provided; normalize-to-C skipped.")

    total_transpose = args.transpose + auto_t
    if total_transpose != 0:
        print(f"Total transpose = {total_transpose} semitones")

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

    events = build_events(melody, total_transpose=total_transpose)
    print(f"events={len(events)}\n")

    play(events, start_delay=args.start_delay, hold_ratio=args.hold, jitter_ms=args.jitter_ms, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
