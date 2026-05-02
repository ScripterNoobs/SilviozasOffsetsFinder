#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import re
import sys
from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple


BytePattern = List[Optional[int]]


@dataclasses.dataclass
class PatchLine:
    raw: str
    kind: str
    pattern: BytePattern = dataclasses.field(default_factory=list)
    bytes_text: str = ""
    reference_offset: Optional[int] = None


@dataclasses.dataclass
class Strategy:
    name: str
    pattern: BytePattern
    result_delta: int
    priority: int


@dataclasses.dataclass
class Candidate:
    offset: int
    hit: int
    strategy: str
    priority: int
    candidate_count: int


@dataclasses.dataclass
class InferredOffset:
    offset: int
    source: int
    strategy: str
    delta: int


OFFSET_RE = re.compile(r"^0[xX]([0-9a-fA-F]+)\s+(.+)$")


def trim(line: str) -> str:
    return line.strip(" \t\r\n")


def is_wildcard_char(ch: str) -> bool:
    return ch in "?."


def parse_hex_pattern(text: str) -> BytePattern:
    pattern: BytePattern = []
    tokens = re.findall(r"\S+", text)

    for token in tokens:
        i = 0
        while i < len(token):
            c1 = token[i]
            c2 = token[i + 1] if i + 1 < len(token) else ""

            if is_wildcard_char(c1) and (not c2 or is_wildcard_char(c2)):
                pattern.append(None)
                i += 2 if c2 else 1
            elif not is_wildcard_char(c1) and c2 and not is_wildcard_char(c2):
                try:
                    pattern.append(int(token[i : i + 2], 16))
                except ValueError as exc:
                    raise ValueError(f"invalid hex byte: {token[i:i + 2]}") from exc
                i += 2
            elif is_wildcard_char(c1) and c2 and not is_wildcard_char(c2):
                raise ValueError("half wildcard like '?5' is not supported; use '??'")
            elif not is_wildcard_char(c1) and is_wildcard_char(c2):
                raise ValueError("half wildcard like '5?' is not supported; use '??'")
            else:
                raise ValueError("single hex character is not a full byte")

    return pattern


def pattern_key(pattern: Sequence[Optional[int]]) -> str:
    return " ".join("??" if b is None else f"{b:02X}" for b in pattern)


def format_offset(offset: int) -> str:
    return f"0x{offset:06X}"


def split_offset_line(line: str) -> Tuple[Optional[int], str]:
    match = OFFSET_RE.match(line)
    if not match:
        return None, line
    return int(match.group(1), 16), trim(match.group(2))


def read_patch_file(path: pathlib.Path) -> List[PatchLine]:
    lines: List[PatchLine] = []
    for raw in path.read_text(errors="replace").splitlines():
        line = trim(raw)
        if not line:
            lines.append(PatchLine(raw=raw, kind="blank"))
            continue
        if line.startswith("#") or line.startswith(";"):
            lines.append(PatchLine(raw=raw, kind="comment"))
            continue
        if line.startswith("[") or line.startswith("AddVersionSupport"):
            lines.append(PatchLine(raw=raw, kind="header"))
            continue

        ref_offset, bytes_text = split_offset_line(line)
        try:
            pattern = parse_hex_pattern(bytes_text)
        except ValueError:
            lines.append(PatchLine(raw=raw, kind="parse_error", bytes_text=bytes_text))
            continue

        lines.append(
            PatchLine(
                raw=raw,
                kind="pattern",
                pattern=pattern,
                bytes_text=pattern_key(pattern),
                reference_offset=ref_offset,
            )
        )
    return lines


def load_completed_offsets(path: pathlib.Path) -> Dict[str, List[int]]:
    refs: DefaultDict[str, List[int]] = defaultdict(list)
    for line in read_patch_file(path):
        if line.kind == "pattern" and line.reference_offset is not None:
            refs[pattern_key(line.pattern)].append(line.reference_offset)
    return dict(refs)


def wildcard_range(pattern: BytePattern, start: int, count: int) -> None:
    for index in range(start, min(len(pattern), start + count)):
        pattern[index] = None


def fixed(pattern: Sequence[Optional[int]], index: int, value: int) -> bool:
    return 0 <= index < len(pattern) and pattern[index] == value


def make_relaxed_patch_pattern(pattern: Sequence[Optional[int]]) -> BytePattern:
    relaxed = list(pattern)
    i = 0
    while i < len(relaxed):
        value = relaxed[i]
        if value is None:
            i += 1
            continue

        # Short conditional jump opcode can change when a patch inverts logic.
        if 0x70 <= value <= 0x7F:
            wildcard_range(relaxed, i, 1)

        # call/jmp rel32: E8/E9 xx xx xx xx
        if value in (0xE8, 0xE9) and i + 4 < len(relaxed):
            wildcard_range(relaxed, i + 1, 4)
            i += 5
            continue

        # 0F 8x rel32
        if fixed(relaxed, i, 0x0F) and i + 5 < len(relaxed):
            next_value = relaxed[i + 1]
            if next_value is not None and 0x80 <= next_value <= 0x8F:
                wildcard_range(relaxed, i + 2, 4)
                i += 6
                continue

        # RIP-relative disp32: 8B 05 / 89 05 / C6 05 xx xx xx xx
        if (
            i + 5 < len(relaxed)
            and relaxed[i] in (0x8B, 0x89, 0xC6)
            and relaxed[i + 1] == 0x05
        ):
            wildcard_range(relaxed, i + 2, 4)
            i += 6
            continue

        # REX + RIP-relative disp32.
        if (
            i + 6 < len(relaxed)
            and relaxed[i] in (0x48, 0x4C, 0x44)
            and relaxed[i + 1] in (0x8B, 0x8D, 0x89)
            and relaxed[i + 2] is not None
            and (relaxed[i + 2] & 0xC7) == 0x05
        ):
            wildcard_range(relaxed, i + 3, 4)
            i += 7
            continue

        # x64 stack frame disp32: 48 8B/89/8D 84 24 xx xx xx xx
        if (
            i + 7 < len(relaxed)
            and relaxed[i] in (0x48, 0x4C)
            and relaxed[i + 1] in (0x8B, 0x89, 0x8D)
            and relaxed[i + 2] == 0x84
            and relaxed[i + 3] == 0x24
        ):
            wildcard_range(relaxed, i + 4, 4)
            i += 8
            continue

        i += 1
    return relaxed


def mask_window_volatiles(pattern: BytePattern) -> BytePattern:
    masked = list(pattern)
    i = 0
    while i < len(masked):
        value = masked[i]
        if value is None:
            i += 1
            continue

        if value in (0xE8, 0xE9) and i + 4 < len(masked):
            wildcard_range(masked, i + 1, 4)
            i += 5
            continue

        if 0x70 <= value <= 0x7F and i + 1 < len(masked):
            wildcard_range(masked, i + 1, 1)
            i += 2
            continue

        if fixed(masked, i, 0x0F) and i + 5 < len(masked):
            next_value = masked[i + 1]
            if next_value is not None and 0x80 <= next_value <= 0x8F:
                wildcard_range(masked, i + 2, 4)
                i += 6
                continue

        if (
            i + 5 < len(masked)
            and masked[i] in (0x8B, 0x89, 0xC6, 0xC7)
            and masked[i + 1] == 0x05
        ):
            wildcard_range(masked, i + 2, 4)
            i += 6
            continue

        if (
            i + 6 < len(masked)
            and masked[i] in (0x48, 0x4C, 0x44)
            and masked[i + 1] in (0x8B, 0x8D, 0x89)
            and masked[i + 2] is not None
            and (masked[i + 2] & 0xC7) == 0x05
        ):
            wildcard_range(masked, i + 3, 4)
            i += 7
            continue

        if (
            i + 7 < len(masked)
            and masked[i] in (0x48, 0x4C)
            and masked[i + 1] in (0x8B, 0x89, 0x8D)
            and masked[i + 2] == 0x84
            and masked[i + 3] == 0x24
        ):
            wildcard_range(masked, i + 4, 4)
            i += 8
            continue

        if (
            i + 6 < len(masked)
            and masked[i] in (0x8B, 0x89, 0xC6, 0xC7, 0x0F)
            and masked[i + 1] is not None
            and masked[i + 2] == 0x80
        ):
            wildcard_range(masked, i + 3, 4)
            i += 7
            continue

        i += 1
    return masked


def fixed_count(pattern: Sequence[Optional[int]]) -> int:
    return sum(1 for b in pattern if b is not None)


def find_all(data: bytes, pattern: Sequence[Optional[int]]) -> List[int]:
    if not pattern or len(pattern) > len(data):
        return []

    if all(b is not None for b in pattern):
        needle = bytes(b for b in pattern if b is not None)
        hits: List[int] = []
        start = 0
        while True:
            index = data.find(needle, start)
            if index < 0:
                return hits
            hits.append(index)
            start = index + 1

    anchor_index = -1
    anchor_value = -1
    for index, value in enumerate(pattern):
        if value is not None:
            anchor_index = index
            anchor_value = value
            break
    if anchor_index < 0:
        return []

    hits = []
    start = 0
    max_start = len(data) - len(pattern)
    while True:
        pos = data.find(bytes([anchor_value]), start)
        if pos < 0:
            return hits
        candidate = pos - anchor_index
        if 0 <= candidate <= max_start:
            ok = True
            for index, value in enumerate(pattern):
                if value is not None and data[candidate + index] != value:
                    ok = False
                    break
            if ok:
                hits.append(candidate)
        start = pos + 1


def nearby_hits(hits: Sequence[int], offset: int, radius: int) -> List[int]:
    return [hit for hit in hits if abs(hit - offset) <= radius]


def window_strategies(reference_data: bytes, offset: int) -> List[Strategy]:
    strategies: List[Strategy] = []
    seen = set()
    layouts = [
        (64, 64),
        (48, 48),
        (32, 64),
        (24, 72),
        (16, 80),
        (32, 32),
        (16, 48),
        (8, 40),
        (0, 48),
        (0, 32),
        (0, 24),
        (0, 16),
    ]

    for before, after in layouts:
        start = max(0, offset - before)
        end = min(len(reference_data), offset + after)
        if end <= start:
            continue
        prefix = offset - start
        exact = list(reference_data[start:end])
        for name, pattern, priority in (
            (f"window-{before}+{after}", exact, 0),
            (f"window-masked-{before}+{after}", mask_window_volatiles(exact), 1),
        ):
            if fixed_count(pattern) < 10:
                continue
            key = (name, pattern_key(pattern), prefix)
            if key in seen:
                continue
            seen.add(key)
            strategies.append(Strategy(name=name, pattern=pattern, result_delta=prefix, priority=priority))
    return strategies


def patch_anchor_strategies(
    reference_data: bytes,
    patch_pattern: Sequence[Optional[int]],
    reference_offset: int,
) -> List[Strategy]:
    strategies: List[Strategy] = []
    variants = [
        ("patch-anchor", list(patch_pattern), 2),
        ("patch-anchor-relaxed", make_relaxed_patch_pattern(patch_pattern), 3),
    ]

    seen = set()
    for name, pattern, priority in variants:
        if fixed_count(pattern) < 4:
            continue
        hits = nearby_hits(find_all(reference_data, pattern), reference_offset, 0x3000)
        for hit in sorted(hits, key=lambda item: abs(item - reference_offset))[:8]:
            delta = reference_offset - hit
            key = (name, pattern_key(pattern), delta)
            if key in seen:
                continue
            seen.add(key)
            strategies.append(Strategy(name=f"{name}@{delta:+X}", pattern=pattern, result_delta=delta, priority=priority))
    return strategies


def fallback_strategies(pattern: Sequence[Optional[int]]) -> List[Strategy]:
    relaxed = make_relaxed_patch_pattern(pattern)
    strategies = [Strategy(name="patch-bytes", pattern=list(pattern), result_delta=0, priority=5)]
    if pattern_key(relaxed) != pattern_key(pattern):
        strategies.append(Strategy(name="patch-bytes-relaxed", pattern=relaxed, result_delta=0, priority=6))
    return strategies


def direct_match_offsets(data: bytes, pattern: Sequence[Optional[int]]) -> Tuple[List[int], str]:
    exact = find_all(data, pattern)
    if exact:
        return exact, "exact"

    relaxed = make_relaxed_patch_pattern(pattern)
    if pattern_key(relaxed) == pattern_key(pattern):
        return [], "exact"

    return find_all(data, relaxed), "relaxed"


def collect_candidates(target_data: bytes, strategies: Sequence[Strategy]) -> List[Candidate]:
    candidates: List[Candidate] = []
    seen = set()
    for strategy in strategies:
        hits = find_all(target_data, strategy.pattern)
        for hit in hits:
            offset = hit + strategy.result_delta
            if offset < 0 or offset >= len(target_data):
                continue
            key = (offset, strategy.name)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                Candidate(
                    offset=offset,
                    hit=hit,
                    strategy=strategy.name,
                    priority=strategy.priority,
                    candidate_count=len(hits),
                )
            )
    return candidates


def choose_candidate(
    candidates: Sequence[Candidate],
    used_offsets: Sequence[int],
    reference_offset: Optional[int],
    previous_reference: Optional[int],
    previous_found: Optional[int],
    same_reference_file: bool,
) -> Optional[Candidate]:
    if not candidates:
        return None

    used = set(used_offsets)
    available = [candidate for candidate in candidates if candidate.offset not in used]
    if not available:
        available = list(candidates)

    expected: Optional[int] = None
    if previous_reference is not None and previous_found is not None and reference_offset is not None:
        expected = previous_found + (reference_offset - previous_reference)
    elif same_reference_file and reference_offset is not None:
        expected = reference_offset

    def score(candidate: Candidate) -> Tuple[int, int, int, int]:
        distance = abs(candidate.offset - expected) if expected is not None else 0
        ref_distance = abs(candidate.offset - reference_offset) if reference_offset is not None else 0
        return (candidate.priority, distance, ref_distance, candidate.offset)

    return min(available, key=score)


def auto_completed_path(patch_path: pathlib.Path) -> Optional[pathlib.Path]:
    candidates = [
        patch_path.with_name("PatchDataCompleted"),
        pathlib.Path.cwd() / "PatchDataCompleted",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def resolve_reference_for_line(
    line: PatchLine,
    completed_offsets: Dict[str, List[int]],
    completed_cursors: Dict[str, int],
) -> Optional[int]:
    if line.reference_offset is not None:
        return line.reference_offset

    key = pattern_key(line.pattern)
    refs = completed_offsets.get(key)
    if not refs:
        return None

    cursor = completed_cursors.get(key, 0)
    if cursor >= len(refs):
        return None

    completed_cursors[key] = cursor + 1
    return refs[cursor]


def pattern_i32(pattern: Sequence[Optional[int]], offset: int) -> Optional[int]:
    if offset + 4 > len(pattern):
        return None
    values = pattern[offset : offset + 4]
    if any(value is None for value in values):
        return None
    return int.from_bytes(bytes(value for value in values if value is not None), "little", signed=True)


def data_i32(data: bytes, offset: int) -> Optional[int]:
    if offset < 0 or offset + 4 > len(data):
        return None
    return int.from_bytes(data[offset : offset + 4], "little", signed=True)


def is_counter_patch_pattern(pattern: Sequence[Optional[int]]) -> bool:
    if len(pattern) < 16:
        return False
    required = {
        0: 0x75,
        1: 0x1A,
        2: 0x8B,
        3: 0x05,
        8: 0xFF,
        9: 0xC0,
        10: 0x89,
        11: 0x05,
    }
    for index, value in required.items():
        if pattern[index] != value:
            return False
    disp1 = pattern_i32(pattern, 4)
    disp2 = pattern_i32(pattern, 12)
    return disp1 is not None and disp2 is not None and disp2 == disp1 - 8


def counter_source_abs(data: bytes, hit: int) -> Optional[int]:
    if hit < 0 or hit + 16 > len(data):
        return None
    if not (
        data[hit] == 0x75
        and data[hit + 1] == 0x1A
        and data[hit + 2] == 0x8B
        and data[hit + 3] == 0x05
        and data[hit + 8] == 0xFF
        and data[hit + 9] == 0xC0
        and data[hit + 10] == 0x89
        and data[hit + 11] == 0x05
    ):
        return None
    disp1 = data_i32(data, hit + 4)
    disp2 = data_i32(data, hit + 12)
    if disp1 is None or disp2 is None:
        return None
    abs1 = hit + 8 + disp1
    abs2 = hit + 16 + disp2
    if abs1 != abs2:
        return None
    return abs1


def align_up(value: int, alignment: int) -> int:
    if value <= 0:
        return 0
    return ((value + alignment - 1) // alignment) * alignment


def grouped_counter_indices(pattern_items: Sequence[Tuple[int, PatchLine]]) -> Tuple[int, List[int]]:
    counter_positions = [
        pattern_pos
        for pattern_pos, (_, line) in enumerate(pattern_items)
        if is_counter_patch_pattern(line.pattern)
    ]
    if len(counter_positions) < 2:
        return 0, counter_positions

    diffs = [
        b - a
        for a, b in zip(counter_positions, counter_positions[1:])
        if b > a
    ]
    if not diffs:
        return 0, counter_positions

    group_size = max(set(diffs), key=diffs.count)
    return group_size, counter_positions


def cached_hits_for(
    data: bytes,
    pattern: Sequence[Optional[int]],
    cache: Dict[str, List[int]],
    relaxed: bool = False,
) -> List[int]:
    actual_pattern = make_relaxed_patch_pattern(pattern) if relaxed else list(pattern)
    key = ("R:" if relaxed else "S:") + pattern_key(actual_pattern)
    if key not in cache:
        cache[key] = find_all(data, actual_pattern)
    return cache[key]


def choose_previous_source(
    hits: Sequence[int],
    counter_source: int,
    occurrence: int,
) -> Optional[int]:
    before = [hit for hit in hits if hit < counter_source]
    if before:
        return before[-1]
    if occurrence < len(hits):
        return hits[occurrence]
    return None


def common_positive_relatives(counter_sources: Sequence[int], hits: Sequence[int], limit: int = 0x10000) -> List[int]:
    common: Optional[set[int]] = None
    hit_set = set(hits)
    for source in counter_sources:
        relatives = {
            hit - source
            for hit in hit_set
            if 0 < hit - source < limit
        }
        common = relatives if common is None else common & relatives
    return sorted(common or [])


def branch_rel32_at(data: bytes, hit: int) -> int:
    if hit + 7 > len(data):
        return 0x7FFFFFFF
    return int.from_bytes(data[hit + 3 : hit + 7], "little", signed=True)


def choose_following_relative(
    data: bytes,
    pattern: Sequence[Optional[int]],
    counter_sources: Sequence[int],
    hits: Sequence[int],
) -> Optional[int]:
    relatives = common_positive_relatives(counter_sources, hits)
    if not relatives:
        return None

    wanted_first = pattern[0] if pattern else None
    same_first = [
        rel
        for rel in relatives
        if wanted_first is None or data[counter_sources[0] + rel] == wanted_first
    ]
    candidates = same_first or relatives

    # For the Silviozas branch patch layout, several nearby Jcc/JMP snippets
    # share the same skeleton. The one to clone is the latest same-opcode branch
    # with the shortest remaining jump, i.e. the final matching gate in the run.
    if len(pattern) >= 7 and pattern[1:3] == [0x05, 0xE9]:
        return min(candidates, key=lambda rel: (branch_rel32_at(data, counter_sources[0] + rel), -rel))

    return candidates[0]


def infer_offsets_without_completed(
    data: bytes,
    patch_lines: Sequence[PatchLine],
    expected_copies: int = 4,
    verbose: bool = False,
) -> Dict[int, InferredOffset]:
    pattern_items = [
        (line_index, line)
        for line_index, line in enumerate(patch_lines)
        if line.kind == "pattern"
    ]
    group_size, counter_positions = grouped_counter_indices(pattern_items)
    if group_size <= 0 or not counter_positions:
        return {}

    hit_cache: Dict[str, List[int]] = {}
    counter_sources: List[int] = []
    counter_base_deltas: List[int] = []
    counter_line_positions: List[int] = []

    for occurrence, pattern_pos in enumerate(counter_positions):
        line_index, line = pattern_items[pattern_pos]
        relaxed_hits = cached_hits_for(data, line.pattern, hit_cache, relaxed=True)
        valid_hits = [hit for hit in relaxed_hits if counter_source_abs(data, hit) is not None]
        if occurrence >= len(valid_hits):
            continue

        source = valid_hits[occurrence]
        actual_disp = data_i32(data, source + 4)
        patch_disp = pattern_i32(line.pattern, 4)
        if actual_disp is None or patch_disp is None:
            continue

        counter_sources.append(source)
        counter_base_deltas.append(actual_disp - patch_disp)
        counter_line_positions.append(pattern_pos)

    if not counter_sources:
        return {}
    if verbose and expected_copies > 0 and len(counter_sources) != expected_copies:
        print(f"[!] Expected {expected_copies} copies, found {len(counter_sources)} counter copies")

    global_shift = align_up(-min(counter_base_deltas), 0x1000)
    deltas = [base_delta + global_shift for base_delta in counter_base_deltas]
    inferred: Dict[int, InferredOffset] = {}

    if verbose:
        abs_values = [counter_source_abs(data, source) for source in counter_sources]
        abs_text = ", ".join(format_offset(value) for value in abs_values if value is not None)
        print(
            f"[*] Counter inference: group={group_size}, copies={len(counter_sources)}, "
            f"shift={format_offset(global_shift)}, abs=[{abs_text}]"
        )

    for occurrence, (pattern_pos, source, delta) in enumerate(zip(counter_line_positions, counter_sources, deltas)):
        line_index, _ = pattern_items[pattern_pos]
        inferred[line_index] = InferredOffset(
            offset=source + delta,
            source=source,
            strategy="counter-rip",
            delta=delta,
        )

        group_start = pattern_pos - (pattern_pos % group_size)
        group_end = min(group_start + group_size, len(pattern_items))

        # Lines before the RIP-counter patch use the same per-group shift.
        for previous_pos in range(group_start, pattern_pos):
            previous_line_index, previous_line = pattern_items[previous_pos]
            hits = cached_hits_for(data, previous_line.pattern, hit_cache, relaxed=False)
            if not hits:
                hits = cached_hits_for(data, previous_line.pattern, hit_cache, relaxed=True)
            previous_source = choose_previous_source(hits, source, occurrence)
            if previous_source is None:
                continue
            inferred[previous_line_index] = InferredOffset(
                offset=previous_source + delta,
                source=previous_source,
                strategy="group-shift-before-counter",
                delta=delta,
            )

        # Lines after the RIP-counter patch are selected by common relative
        # branch anchors inside every repeated group.
        for next_pos in range(pattern_pos + 1, group_end):
            next_line_index, next_line = pattern_items[next_pos]
            hits = cached_hits_for(data, next_line.pattern, hit_cache, relaxed=True)
            relative = choose_following_relative(data, next_line.pattern, counter_sources, hits)
            if relative is None:
                continue

            source_anchor = source + relative
            lead_in = 0
            if len(next_line.pattern) >= 7 and next_line.pattern[0:3] == [0x75, 0x05, 0xE9]:
                lead_in = 0x54

            inferred[next_line_index] = InferredOffset(
                offset=source_anchor + delta - lead_in,
                source=source_anchor,
                strategy=f"group-branch-anchor-{relative:+X}",
                delta=delta - lead_in,
            )

    return inferred


def run(args: argparse.Namespace) -> int:
    input_path = pathlib.Path(args.input)
    patch_path = pathlib.Path(args.patchdata)
    output_path = pathlib.Path(args.output)

    target_data = input_path.read_bytes()
    patch_lines = read_patch_file(patch_path)

    completed_path: Optional[pathlib.Path] = pathlib.Path(args.completed) if args.completed else None

    completed_offsets: Dict[str, List[int]] = {}
    if completed_path is not None and completed_path.is_file():
        completed_offsets = load_completed_offsets(completed_path)
    inferred_offsets: Dict[int, InferredOffset] = {}
    if not completed_offsets:
        inferred_offsets = infer_offsets_without_completed(target_data, patch_lines, args.copies, args.verbose)

    reference_path = pathlib.Path(args.reference_exe) if args.reference_exe else input_path
    reference_data = reference_path.read_bytes()
    same_reference_file = input_path.resolve() == reference_path.resolve()

    if args.verbose:
        print(f"[*] Input EXE      : {input_path}")
        print(f"[*] PatchData      : {patch_path}")
        print(f"[*] Output         : {output_path}")
        print(f"[*] Reference EXE  : {reference_path}")
        if completed_path is not None and completed_path.is_file():
            print(f"[*] Completed refs : {completed_path} ({len(completed_offsets)} pattern)")
        elif inferred_offsets:
            print(
                f"[*] Completed refs : disabled; inferred {len(inferred_offsets)} offsets "
                f"from PatchData ({args.copies} copies expected)"
            )
        print(f"[*] EXE size       : {len(target_data)} byte")

    output_lines: List[str] = []
    completed_cursors: Dict[str, int] = {}
    used_offsets: List[int] = []
    previous_reference: Optional[int] = None
    previous_found: Optional[int] = None
    total_patterns = 0
    total_found = 0
    total_missing = 0

    for index, line in enumerate(patch_lines, start=1):
        line_index = index - 1
        if line.kind == "blank":
            output_lines.append("")
            continue
        if line.kind in ("comment", "header"):
            output_lines.append(trim(line.raw))
            if trim(line.raw).startswith("[START_"):
                used_offsets = []
                previous_reference = None
                previous_found = None
            continue
        if line.kind == "parse_error":
            total_missing += 1
            output_lines.append(f"# PARSE_ERROR  {trim(line.raw)}")
            if args.verbose:
                print(f"[-] Line {index}: parse error")
            continue

        total_patterns += 1

        inferred = inferred_offsets.get(line_index)
        if inferred is not None:
            total_found += 1
            found_offset = inferred.offset if args.patch_targets else inferred.source
            output_lines.append(f"{format_offset(found_offset)}  {line.bytes_text}")
            if args.verbose:
                mode_text = "patch target" if args.patch_targets else "hex match"
                print(
                    f"[+] Line {index}: {format_offset(found_offset)}"
                    f" ({mode_text}, {inferred.strategy}, source {format_offset(inferred.source)}, "
                    f"patch {format_offset(inferred.offset)}, delta {inferred.delta:+X})"
                )
            continue

        if not args.patch_targets and not completed_offsets:
            matches, match_kind = direct_match_offsets(target_data, line.pattern)
            if matches:
                total_found += len(matches)
                for match_offset in matches:
                    output_lines.append(f"{format_offset(match_offset)}  {line.bytes_text}")
                if args.verbose:
                    print(f"[+] Line {index}: {len(matches)} {match_kind} match(es)")
                continue

        reference_offset = resolve_reference_for_line(line, completed_offsets, completed_cursors)

        strategies: List[Strategy] = []
        if reference_offset is not None and 0 <= reference_offset < len(reference_data):
            strategies.extend(window_strategies(reference_data, reference_offset))
            strategies.extend(patch_anchor_strategies(reference_data, line.pattern, reference_offset))

        if not strategies or args.allow_fallback:
            strategies.extend(fallback_strategies(line.pattern))

        candidates = collect_candidates(target_data, strategies)
        selected = choose_candidate(
            candidates=candidates,
            used_offsets=used_offsets,
            reference_offset=reference_offset,
            previous_reference=previous_reference,
            previous_found=previous_found,
            same_reference_file=same_reference_file,
        )

        if selected is None:
            total_missing += 1
            output_lines.append(f"# NOT_FOUND  {line.bytes_text}")
            if args.verbose:
                print(f"[-] Line {index}: not found")
            continue

        total_found += 1
        used_offsets.append(selected.offset)
        previous_found = selected.offset
        previous_reference = reference_offset if reference_offset is not None else selected.offset
        output_lines.append(f"{format_offset(selected.offset)}  {line.bytes_text}")

        if args.verbose:
            ref_text = f", ref {format_offset(reference_offset)}" if reference_offset is not None else ""
            print(
                f"[+] Line {index}: {format_offset(selected.offset)}"
                f" ({selected.strategy}, {selected.candidate_count} hit{ref_text})"
            )

    output_path.write_text("\n".join(output_lines) + "\n")

    if args.verbose:
        print()
        print(f"[=] Total pattern : {total_patterns}")
        print(f"[=] Found         : {total_found}")
        print(f"[=] Missing       : {total_missing}")
        print(f"[=] Output        : {output_path}")

    return 0 if total_missing == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find AutoCrack patch offsets.")
    parser.add_argument("-i", "--input", required=True, help="target EXE to scan")
    parser.add_argument("-m", "--patchdata", required=True, help="PatchData or PatchDataCompleted input")
    parser.add_argument("-o", "--output", required=True, help="output PatchData file with offsets")
    parser.add_argument(
        "-c",
        "--completed",
        help="optional PatchDataCompleted reference file; disabled unless explicitly passed",
    )
    parser.add_argument(
        "--reference-exe",
        help="EXE that matches PatchDataCompleted; defaults to --input",
    )
    parser.add_argument(
        "--no-fallback",
        dest="allow_fallback",
        action="store_false",
        help="disable direct patch-byte fallback searches",
    )
    parser.add_argument(
        "--copies",
        type=int,
        default=4,
        help="expected repeated copies for each PatchData pattern family (default: 4)",
    )
    parser.add_argument(
        "--patch-targets",
        action="store_true",
        help="write patch target offsets instead of the offsets where the hex patterns are found",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="print match details")
    parser.set_defaults(allow_fallback=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except FileNotFoundError as exc:
        print(f"[!] File not found: {exc.filename}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"[!] I/O error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
