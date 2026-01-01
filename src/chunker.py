"""Content chunking for better search granularity.

Supports language-aware chunking that respects code structure (functions,
classes, etc.) when a language is detected, falling back to blank-line
splitting for unknown languages.
"""

from dataclasses import dataclass

from .languages import LanguageSpec


@dataclass
class Chunk:
    """A chunk of file content with line position info."""

    start_line: int  # 1-indexed
    end_line: int  # 1-indexed, inclusive
    content: str


def chunk_content(
    content: str,
    language: LanguageSpec | None = None,
    min_lines: int = 10,
    max_lines: int = 50,
    overlap: int = 5,
) -> list[Chunk]:
    """Split content into chunks using language-aware boundaries.

    Strategy:
    1. If language is known, recursively split using language-specific
       separators (class, function definitions, etc.)
    2. If language is unknown, split on blank lines (natural boundaries)
    3. Merge consecutive small chunks until they reach min_lines
    4. Split chunks larger than max_lines with overlap

    Args:
        content: The file content to chunk.
        language: Optional LanguageSpec for syntax-aware splitting.
        min_lines: Minimum chunk size (merge smaller chunks).
        max_lines: Maximum chunk size (split larger chunks).
        overlap: Lines of overlap when splitting large chunks.

    Returns:
        List of Chunk objects with content and line positions.
    """
    if not content.strip():
        return []

    # Step 1: Initial split using language separators or blank lines
    if language:
        segments = _split_with_separators(content, language.separators)
    else:
        segments = _split_on_blank_lines(content)

    # Step 2: Merge small consecutive segments
    merged = _merge_small_segments(segments, min_lines)

    # Step 3: Split large segments with overlap
    chunks = _split_large_segments(merged, max_lines, overlap)

    return chunks


def _split_with_separators(content: str, separators: tuple[str, ...]) -> list[Chunk]:
    """Recursively split content using ordered separators.

    Tries each separator in order. If a separator produces splits,
    recursively applies remaining separators to chunks that are still large.
    """
    lines = content.split("\n")

    # Try each separator until one produces a split
    for sep in separators:
        if sep == "\n\n":
            # Special case: blank line splitting (final fallback)
            return _split_on_blank_lines(content)

        parts = content.split(sep)
        if len(parts) > 1:
            # This separator worked - build chunks
            chunks = []
            current_pos = 0

            for i, part in enumerate(parts):
                if i > 0:
                    # Re-add the separator (it's part of the next chunk)
                    part = sep.lstrip("\n") + part

                if not part.strip():
                    # Count lines in empty part for position tracking
                    current_pos += part.count("\n")
                    continue

                # Calculate line positions
                start_line = current_pos + 1
                part_lines = part.count("\n") + 1
                end_line = current_pos + part_lines

                chunks.append(Chunk(
                    start_line=start_line,
                    end_line=end_line,
                    content=part.strip(),
                ))

                current_pos = end_line

            if chunks:
                return chunks

    # No separator worked, return as single chunk
    return [Chunk(
        start_line=1,
        end_line=len(lines),
        content=content.strip(),
    )]


def _split_on_blank_lines(content: str) -> list[Chunk]:
    """Split content into segments at blank line boundaries."""
    lines = content.split("\n")
    segments = []
    current_start = 0
    current_lines: list[str] = []

    for i, line in enumerate(lines):
        if line.strip() == "" and current_lines:
            # End of a segment
            segments.append(Chunk(
                start_line=current_start + 1,  # 1-indexed
                end_line=i,  # 1-indexed (previous line)
                content="\n".join(current_lines),
            ))
            current_lines = []
            current_start = i + 1
        elif line.strip() != "":
            if not current_lines:
                current_start = i
            current_lines.append(line)

    # Don't forget the last segment
    if current_lines:
        segments.append(Chunk(
            start_line=current_start + 1,
            end_line=len(lines),
            content="\n".join(current_lines),
        ))

    return segments


def _merge_small_segments(segments: list[Chunk], min_lines: int) -> list[Chunk]:
    """Merge consecutive small segments until they reach min_lines."""
    if not segments:
        return []

    merged = []
    current = segments[0]

    for next_seg in segments[1:]:
        current_size = current.end_line - current.start_line + 1

        if current_size < min_lines:
            # Merge with next segment
            current = Chunk(
                start_line=current.start_line,
                end_line=next_seg.end_line,
                content=current.content + "\n\n" + next_seg.content,
            )
        else:
            merged.append(current)
            current = next_seg

    merged.append(current)
    return merged


def _split_large_segments(
    segments: list[Chunk], max_lines: int, overlap: int
) -> list[Chunk]:
    """Split segments larger than max_lines with overlap."""
    result = []

    for seg in segments:
        seg_lines = seg.content.split("\n")
        seg_size = len(seg_lines)

        if seg_size <= max_lines:
            result.append(seg)
            continue

        # Split with overlap
        start = 0
        while start < seg_size:
            end = min(start + max_lines, seg_size)
            chunk_lines = seg_lines[start:end]

            result.append(Chunk(
                start_line=seg.start_line + start,
                end_line=seg.start_line + end - 1,
                content="\n".join(chunk_lines),
            ))

            # Advance by (max_lines - overlap), but ensure we make progress
            step = max(max_lines - overlap, 1)
            if start + step >= seg_size:
                break
            start += step

    return result
