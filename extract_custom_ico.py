#!/usr/bin/env python3
import argparse
import struct
import sys
from array import array
from datetime import datetime
from pathlib import Path


def _is_entry_candidate(chunk):
    if len(chunk) < 8:
        return False
    if chunk == b"\x00" * 8:
        return False
    w_h, _fmt, _off_lo, _off_hi = struct.unpack("<HHHH", chunk)
    width = w_h & 0xFF
    height = (w_h >> 8) & 0xFF
    return width != 0 and height != 0


def parse_entries(data, max_header_bytes=0x2000):
    entries = []
    max_offset = min(len(data), max_header_bytes)
    start_offset = 0 if _is_entry_candidate(data[:8]) else 8
    base_index = 0 if start_offset == 0 else 1
    for offset in range(start_offset, max_offset, 8):
        chunk = data[offset:offset + 8]
        if len(chunk) < 8:
            break
        entry_idx = base_index + (offset - start_offset) // 8
        if chunk == b"\x00" * 8:
            entries.append(
                {
                    "index": entry_idx,
                    "width": None,
                    "height": None,
                    "fmt16": None,
                    "off_lo": None,
                    "off_hi": None,
                }
            )
            continue
        w_h, fmt16, _off_lo, _off_hi = struct.unpack("<HHHH", chunk)
        width = w_h & 0xFF
        height = (w_h >> 8) & 0xFF
        if width and height:
            entries.append(
                {
                    "index": entry_idx,
                    "width": width,
                    "height": height,
                    "fmt16": fmt16,
                    "off_lo": _off_lo,
                    "off_hi": _off_hi,
                }
            )
        else:
            entries.append(
                {
                    "index": entry_idx,
                    "width": None,
                    "height": None,
                    "fmt16": None,
                    "off_lo": None,
                    "off_hi": None,
                }
            )
    last = max(
        (i for i, entry in enumerate(entries) if entry["width"] is not None),
        default=-1,
    )
    if last == -1:
        return [], start_offset
    entries = entries[:last + 1]
    header_end = start_offset + len(entries) * 8
    return entries, header_end


def guess_data_offset(data, header_end, expected_len, forced):
    data_len = len(data)
    if forced is not None:
        return forced
    if data_len >= expected_len:
        candidate = data_len - expected_len
        if candidate >= header_end:
            return candidate
    for i in range(header_end, data_len):
        if data_len - i >= expected_len and data[i] != 0:
            return i
    raise ValueError("Could not locate data offset automatically.")


def unpack_words(data, offset, count, swap_bytes):
    buf = data[offset:offset + count * 2]
    words = array("H")
    words.frombytes(buf)
    if sys.byteorder != "little":
        words.byteswap()
    if swap_bytes:
        words.byteswap()
    return words


def byteswap16(value):
    return ((value & 0xFF) << 8) | ((value >> 8) & 0xFF)


def rgb_from_word(word, fmt):
    if fmt == "rgb565":
        r = (word >> 11) & 0x1F
        g = (word >> 5) & 0x3F
        b = word & 0x1F
        r = (r << 3) | (r >> 2)
        g = (g << 2) | (g >> 4)
        b = (b << 3) | (b >> 2)
        return r, g, b
    if fmt == "bgr565":
        b = (word >> 11) & 0x1F
        g = (word >> 5) & 0x3F
        r = word & 0x1F
        r = (r << 3) | (r >> 2)
        g = (g << 2) | (g >> 4)
        b = (b << 3) | (b >> 2)
        return r, g, b
    if fmt == "rgb555":
        r = (word >> 10) & 0x1F
        g = (word >> 5) & 0x1F
        b = word & 0x1F
        r = (r << 3) | (r >> 2)
        g = (g << 3) | (g >> 2)
        b = (b << 3) | (b >> 2)
        return r, g, b
    if fmt == "bgr555":
        b = (word >> 10) & 0x1F
        g = (word >> 5) & 0x1F
        r = word & 0x1F
        r = (r << 3) | (r >> 2)
        g = (g << 3) | (g >> 2)
        b = (b << 3) | (b >> 2)
        return r, g, b
    raise ValueError(f"Unsupported format: {fmt}")


def write_bmp(path, width, height, words, fmt, flip_vertical):
    row_stride = (width * 3 + 3) & ~3
    padding = row_stride - width * 3
    image_size = row_stride * height
    file_size = 14 + 40 + image_size

    with path.open("wb") as f:
        f.write(b"BM")
        f.write(struct.pack("<IHHI", file_size, 0, 0, 54))
        f.write(struct.pack(
            "<IIIHHIIIIII",
            40,
            width,
            height,
            1,
            24,
            0,
            image_size,
            2835,
            2835,
            0,
            0,
        ))

        if flip_vertical:
            row_indices = range(height)
        else:
            row_indices = range(height - 1, -1, -1)

        for y in row_indices:
            row_start = y * width
            row_end = row_start + width
            row = words[row_start:row_end]
            for word in row:
                r, g, b = rgb_from_word(word, fmt)
                f.write(bytes((b, g, r)))
            if padding:
                f.write(b"\x00" * padding)


def smoothness(words, width, height, fmt):
    if len(words) < width * height:
        return None
    score = 0
    for y in range(height):
        row_start = y * width
        row_end = row_start + width - 1
        for i in range(row_start, row_end):
            r1, g1, b1 = rgb_from_word(words[i], fmt)
            r2, g2, b2 = rgb_from_word(words[i + 1], fmt)
            score += abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2)
    for y in range(height - 1):
        row_start = y * width
        row_next = row_start + width
        for i in range(width):
            r1, g1, b1 = rgb_from_word(words[row_start + i], fmt)
            r2, g2, b2 = rgb_from_word(words[row_next + i], fmt)
            score += abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2)
    return score


def order_entries(entries, mode):
    def size(entry):
        return entry["width"] * entry["height"]

    if mode == "index":
        return sorted(entries, key=lambda e: e["index"])
    if mode == "u16-then-index":
        return sorted(entries, key=lambda e: (e["fmt16"], e["index"]))
    if mode == "u16-then-index-300-offhi0-last":
        group_other = sorted(
            (e for e in entries if e["fmt16"] != 0x300),
            key=lambda e: (e["fmt16"], e["index"]),
        )
        group_300_nonzero_offhi = sorted(
            (e for e in entries if e["fmt16"] == 0x300 and (e["off_hi"] or 0) != 0),
            key=lambda e: e["index"],
        )
        group_300_zero_offhi = sorted(
            (e for e in entries if e["fmt16"] == 0x300 and (e["off_hi"] or 0) == 0),
            key=lambda e: e["index"],
        )
        return group_other + group_300_nonzero_offhi + group_300_zero_offhi
    if mode == "u16-then-size":
        return sorted(entries, key=lambda e: (e["fmt16"], size(e)))
    if mode == "u16-then-off-lo":
        return sorted(entries, key=lambda e: (e["fmt16"], e["off_lo"]))
    if mode == "u16-then-off-lo-desc":
        return sorted(entries, key=lambda e: (e["fmt16"], -e["off_lo"]))
    if mode == "size":
        return sorted(entries, key=size)
    if mode == "size-desc":
        return sorted(entries, key=lambda e: -size(e))
    if mode == "off-lo":
        return sorted(entries, key=lambda e: e["off_lo"])
    if mode == "off-lo-desc":
        return sorted(entries, key=lambda e: -e["off_lo"])
    if mode == "off-hi-then-index":
        return sorted(entries, key=lambda e: (e["off_hi"], e["index"]))
    if mode == "off-hi-then-off-lo":
        return sorted(entries, key=lambda e: (e["off_hi"], e["off_lo"]))
    if mode == "off-hi-then-off-lo-desc":
        return sorted(entries, key=lambda e: (e["off_hi"], -e["off_lo"]))
    if mode == "off-hi-then-size":
        return sorted(entries, key=lambda e: (e["off_hi"], size(e)))
    if mode == "u16-index-300-off-lo":
        group_200 = sorted(
            (e for e in entries if e["fmt16"] != 0x300),
            key=lambda e: e["index"],
        )
        group_300 = sorted(
            (e for e in entries if e["fmt16"] == 0x300),
            key=lambda e: e["off_lo"],
        )
        return group_200 + group_300
    if mode == "u16-index-300-off-hi-off-lo":
        group_200 = sorted(
            (e for e in entries if e["fmt16"] != 0x300),
            key=lambda e: e["index"],
        )
        group_300 = sorted(
            (e for e in entries if e["fmt16"] == 0x300),
            key=lambda e: (e["off_hi"], e["off_lo"]),
        )
        return group_200 + group_300
    if mode == "u16-index-300-off-lo-desc":
        group_200 = sorted(
            (e for e in entries if e["fmt16"] != 0x300),
            key=lambda e: e["index"],
        )
        group_300 = sorted(
            (e for e in entries if e["fmt16"] == 0x300),
            key=lambda e: -e["off_lo"],
        )
        return group_200 + group_300
    raise ValueError(f"Unsupported order mode: {mode}")


def maybe_collapse_rare_third_color(words, rare_ratio=0.01):
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
        if len(counts) > 3:
            return words
    if 0 not in counts or len(counts) != 3:
        return words
    nonzero = [(word, count) for word, count in counts.items() if word != 0]
    if len(nonzero) != 2:
        return words
    nonzero.sort(key=lambda item: item[1], reverse=True)
    dominant_word, dominant_count = nonzero[0]
    rare_word, rare_count = nonzero[1]
    if rare_count / len(words) > rare_ratio:
        return words
    for i, word in enumerate(words):
        if word == rare_word:
            words[i] = dominant_word
    return words


def detile_words(words, width, height, tile, tile_order):
    tiles_x = (width + tile - 1) // tile
    tiles_y = (height + tile - 1) // tile
    out = [0] * (width * height)
    idx = 0
    if tile_order == "col":
        tile_iter = ((tx, ty) for tx in range(tiles_x) for ty in range(tiles_y))
    else:
        tile_iter = ((tx, ty) for ty in range(tiles_y) for tx in range(tiles_x))
    for tx, ty in tile_iter:
        for y in range(tile):
            py = ty * tile + y
            if py >= height:
                continue
            row_base = py * width
            for x in range(tile):
                px = tx * tile + x
                if px >= width:
                    continue
                if idx >= len(words):
                    return None
                out[row_base + px] = words[idx]
                idx += 1
    if idx != len(words):
        return None
    return out


def pick_layout(words, width, height, fmt, threshold=0.03, tiles=(4, 8)):
    row_score = smoothness(words, width, height, fmt)
    if row_score is None:
        return words, "row"
    best_words = words
    best_score = row_score
    best_name = "row"
    for tile in tiles:
        for tile_order in ("row", "col"):
            detiled = detile_words(words, width, height, tile, tile_order)
            if detiled is None:
                continue
            score = smoothness(detiled, width, height, fmt)
            if score is None:
                continue
            if score < best_score:
                best_score = score
                best_words = detiled
                best_name = f"tile{tile}-{tile_order}"
    if best_name != "row" and best_score < row_score * (1 - threshold):
        return best_words, best_name
    return words, "row"


def score_order(entries, data, data_offset, fmt):
    cursor = data_offset
    total = 0
    for entry in entries:
        if entry["width"] is None:
            continue
        w = entry["width"]
        h = entry["height"]
        count = w * h
        words_swapped = unpack_words(data, cursor, count, True)
        words_raw = unpack_words(data, cursor, count, False)
        score_swapped = smoothness(words_swapped, w, h, fmt)
        score_raw = smoothness(words_raw, w, h, fmt)
        if score_swapped is None and score_raw is None:
            return None
        if score_raw is None or (
            score_swapped is not None and score_swapped <= score_raw
        ):
            total += score_swapped
        else:
            total += score_raw
        cursor += count * 2
    return total


def extract_file(
    path,
    out_dir,
    fmt,
    data_offset,
    flip_vertical,
    swap_bytes,
    auto_swap,
    offset_mode,
    order_mode,
    auto_order,
    fmt_200,
    fmt_300,
    swap_200,
    swap_300,
    auto_layout,
    collapse_rare_third,
):
    data = path.read_bytes()
    entries, header_end = parse_entries(data)
    if not entries:
        raise ValueError(f"No entries found in {path}")

    image_entries = [entry for entry in entries if entry["width"] is not None]
    if not image_entries:
        raise ValueError(f"No image entries found in {path}")

    expected_len = sum(
        entry["width"] * entry["height"] * 2
        for entry in image_entries
    )
    data_offset = guess_data_offset(data, header_end, expected_len, data_offset)
    if data_offset + expected_len > len(data):
        raise ValueError(f"Data region exceeds file size for {path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    ordered_entries = image_entries
    selected_order_mode = order_mode
    if offset_mode == "sequential":
        if auto_order:
            candidates = (
                "index",
                "u16-then-index",
                "u16-then-index-300-offhi0-last",
                "u16-then-size",
                "u16-then-off-lo",
                "u16-then-off-lo-desc",
                "u16-index-300-off-lo",
                "u16-index-300-off-hi-off-lo",
                "u16-index-300-off-lo-desc",
                "size",
                "size-desc",
                "off-lo",
                "off-lo-desc",
                "off-hi-then-index",
                "off-hi-then-off-lo",
                "off-hi-then-off-lo-desc",
                "off-hi-then-size",
            )
            best_mode = None
            best_score = None
            for mode in candidates:
                order = order_entries(image_entries, mode)
                score = score_order(order, data, data_offset, fmt)
                if score is None:
                    continue
                if best_score is None or score < best_score:
                    best_score = score
                    best_mode = mode
            if best_mode is not None:
                order_mode = best_mode
        selected_order_mode = order_mode
        ordered_entries = order_entries(image_entries, order_mode)
    cursor = data_offset
    extracted = 0
    for entry in ordered_entries:
        w = entry["width"]
        h = entry["height"]
        entry_fmt = fmt_300 if entry["fmt16"] == 0x300 else fmt_200
        entry_swap = swap_300 if entry["fmt16"] == 0x300 else swap_200
        count = w * h

        def pick_words(offset):
            words_swapped = None
            words_raw = None
            score_swapped = None
            score_raw = None
            if auto_swap:
                words_swapped = unpack_words(data, offset, count, True)
                words_raw = unpack_words(data, offset, count, False)
                if words_swapped is not None:
                    score_swapped = smoothness(words_swapped, w, h, entry_fmt)
                if words_raw is not None:
                    score_raw = smoothness(words_raw, w, h, entry_fmt)
                if score_swapped is None and score_raw is None:
                    return None, None
                if score_raw is None or (
                    score_swapped is not None and score_swapped <= score_raw
                ):
                    return words_swapped, score_swapped
                return words_raw, score_raw
            words = unpack_words(data, offset, count, entry_swap)
            if words is None:
                return None, None
            return words, smoothness(words, w, h, entry_fmt)

        if offset_mode == "sequential":
            offset = cursor
            words = unpack_words(data, offset, count, entry_swap)
            if auto_swap:
                words, _ = pick_words(offset)
        elif offset_mode == "banked":
            if entry["fmt16"] is None or entry["off_lo"] is None:
                words = None
            else:
                bank = (entry["fmt16"] >> 8) - 2
                if bank < 0:
                    words = None
                else:
                    start_words = bank * 0x10000 + byteswap16(entry["off_lo"])
                    offset = data_offset + start_words * 2
                    words = unpack_words(data, offset, count, entry_swap)
                    if auto_swap:
                        words, _ = pick_words(offset)
        elif offset_mode in ("off-bytes", "off-words", "off-dwords", "auto"):
            offset_candidates = []
            if entry["off_lo"] is not None:
                off_lo = entry["off_lo"]
                offset_candidates = [
                    ("off-bytes", data_offset + off_lo),
                    ("off-words", data_offset + off_lo * 2),
                    ("off-dwords", data_offset + off_lo * 4),
                    ("off-words-swapped", data_offset + byteswap16(off_lo) * 2),
                ]
                if entry["fmt16"] is not None:
                    bank = (entry["fmt16"] >> 8) - 2
                    if bank >= 0:
                        start_words = bank * 0x10000 + byteswap16(off_lo)
                        offset_candidates.append(
                            ("banked", data_offset + start_words * 2)
                        )
            if offset_mode == "auto":
                best = None
                best_score = None
                for name, candidate in offset_candidates:
                    words_candidate, score = pick_words(candidate)
                    if words_candidate is None or score is None:
                        continue
                    if best_score is None or score < best_score:
                        best = words_candidate
                        best_score = score
                words = best
            else:
                scale = {"off-bytes": 1, "off-words": 2, "off-dwords": 4}[offset_mode]
                offset = data_offset + entry["off_lo"] * scale
                words = unpack_words(data, offset, count, entry_swap)
                if auto_swap:
                    words, _ = pick_words(offset)
        else:
            raise ValueError(f"Unsupported offset mode: {offset_mode}")

        if words is None:
            cursor += count * 2
            continue

        if auto_layout and entry["fmt16"] == 0x300:
            words, _layout = pick_layout(words, w, h, entry_fmt)

        if collapse_rare_third:
            maybe_collapse_rare_third_color(words)

        cursor += count * 2
        out_path = out_dir / f"{entry['index']}.bmp"
        write_bmp(out_path, w, h, words, entry_fmt, flip_vertical)
        extracted += 1

    return {
        "entries": extracted,
        "data_offset": data_offset,
        "expected_len": expected_len,
        "out_dir": out_dir,
        "order_mode": selected_order_mode,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract custom ICO images with raw 16-bit pixels."
    )
    parser.add_argument("files", nargs="+", help="ICO files to extract")
    parser.add_argument(
        "--format",
        dest="fmt",
        choices=("rgb565", "bgr565", "rgb555", "bgr555"),
        default="rgb565",
        help="Pixel format (default: rgb565)",
    )
    parser.add_argument(
        "--fmt-200",
        choices=("rgb565", "bgr565", "rgb555", "bgr555"),
        default=None,
        help="Override format for entries with fmt16 == 0x200",
    )
    parser.add_argument(
        "--fmt-300",
        choices=("rgb565", "bgr565", "rgb555", "bgr555"),
        default=None,
        help="Override format for entries with fmt16 == 0x300",
    )
    parser.add_argument(
        "--out-dir",
        default="extracted",
        help="Output root directory (default: extracted)",
    )
    parser.add_argument(
        "--plain-dir",
        action="store_true",
        help="Use only the ICO stem for output folder names (default behavior)",
    )
    parser.add_argument(
        "--decorated-dir",
        action="store_false",
        dest="plain_dir",
        help="Include format/swap suffix in output folder names",
    )
    parser.add_argument(
        "--no-timestamp-dir",
        action="store_true",
        help="Do not create a timestamped subfolder under the output directory",
    )
    parser.add_argument(
        "--data-offset",
        type=lambda v: int(v, 0),
        default=None,
        help="Override data offset (e.g. 0x40000)",
    )
    parser.add_argument(
        "--flip-vertical",
        action="store_true",
        help="Flip vertically if output appears upside down",
    )
    parser.add_argument(
        "--no-swap-bytes",
        action="store_false",
        dest="swap_bytes",
        help="Disable swapping bytes in each 16-bit word",
    )
    parser.add_argument(
        "--no-swap-200",
        action="store_false",
        dest="swap_200",
        help="Disable swapping bytes for fmt16 == 0x200 entries",
    )
    parser.add_argument(
        "--no-swap-300",
        action="store_false",
        dest="swap_300",
        help="Disable swapping bytes for fmt16 == 0x300 entries",
    )
    parser.add_argument(
        "--auto-swap",
        action="store_true",
        help="Pick swap-bytes per image using a smoothness heuristic",
    )
    parser.add_argument(
        "--offset-mode",
        choices=("banked", "sequential", "off-bytes", "off-words", "off-dwords", "auto"),
        default="banked",
        help="How to choose image offsets (default: banked)",
    )
    parser.add_argument(
        "--order-mode",
        choices=(
            "index",
            "u16-then-index",
            "u16-then-index-300-offhi0-last",
            "u16-then-size",
            "u16-then-off-lo",
            "u16-then-off-lo-desc",
            "u16-index-300-off-lo",
            "u16-index-300-off-hi-off-lo",
            "u16-index-300-off-lo-desc",
            "size",
            "size-desc",
            "off-lo",
            "off-lo-desc",
            "off-hi-then-index",
            "off-hi-then-off-lo",
            "off-hi-then-off-lo-desc",
            "off-hi-then-size",
        ),
        default="index",
        help="How to order images in the data stream (default: index)",
    )
    parser.add_argument(
        "--auto-order",
        action="store_true",
        help="Pick order-mode by testing several candidates",
    )
    parser.add_argument(
        "--auto-layout",
        action="store_true",
        dest="auto_layout",
        help="Enable automatic tile-layout detection (may help for some files)",
    )
    parser.add_argument(
        "--no-auto-layout",
        action="store_false",
        dest="auto_layout",
        help="Disable automatic tile-layout detection",
    )
    parser.add_argument(
        "--collapse-rare-third-color",
        action="store_true",
        help="If an image has exactly 3 colors (including 0) and one is very rare, "
             "replace the rare color with the dominant one",
    )
    parser.set_defaults(
        swap_bytes=True,
        swap_200=None,
        swap_300=None,
        auto_layout=False,
        plain_dir=True,
    )
    args = parser.parse_args()

    fmt_200 = args.fmt_200 or args.fmt
    fmt_300 = args.fmt_300 or args.fmt
    swap_200 = args.swap_bytes if args.swap_200 is None else args.swap_200
    swap_300 = args.swap_bytes if args.swap_300 is None else args.swap_300

    root = Path(args.out_dir)
    if not args.no_timestamp_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = root / timestamp
    for file_path in args.files:
        path = Path(file_path)
        if args.plain_dir:
            out_dir = root / path.stem
        else:
            suffix = args.fmt + ("_swap" if args.swap_bytes else "")
            out_dir = root / f"{path.stem}_{suffix}"
        info = extract_file(
            path,
            out_dir,
            args.fmt,
            args.data_offset,
            args.flip_vertical,
            args.swap_bytes,
            args.auto_swap,
            args.offset_mode,
            args.order_mode,
            args.auto_order,
            fmt_200,
            fmt_300,
            swap_200,
            swap_300,
            args.auto_layout,
            args.collapse_rare_third_color,
        )
        print(
            f"{path.name}: {info['entries']} icons, "
            f"data_offset=0x{info['data_offset']:X}, "
            f"order={info['order_mode']}, "
            f"out={info['out_dir']}"
        )


if __name__ == "__main__":
    main()
