"""Content chunking for better search granularity.

Supports tree-sitter AST-based chunking that respects code structure (functions,
classes, etc.) when a language is detected, falling back to blank-line
splitting for unknown languages or parse failures.
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
    1. If language has a tree-sitter grammar, parse and split on AST node boundaries
    2. Otherwise, split on blank lines (natural boundaries)
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

    # Step 1: Initial split using tree-sitter AST or blank lines
    if language and language.ts_language and language.node_types:
        segments = _split_with_treesitter(content, language)
        if segments is None:
            # Parse failed, fall back
            segments = _split_on_blank_lines(content)
    else:
        segments = _split_on_blank_lines(content)

    # Step 2: Merge small consecutive segments
    merged = _merge_small_segments(segments, min_lines)

    # Step 3: Split large segments with overlap
    chunks = _split_large_segments(merged, max_lines, overlap)

    return chunks


def _split_with_treesitter(content: str, language: LanguageSpec) -> list[Chunk] | None:
    """Split content using tree-sitter AST node boundaries.

    Returns None if parsing fails (caller should fall back to blank-line splitting).
    """
    try:
        from tree_sitter_language_pack import get_parser
    except ImportError:
        return None

    try:
        parser = get_parser(language.ts_language)
    except Exception:
        return None

    tree = parser.parse(content.encode("utf-8"))
    root = tree.root_node

    if root.child_count == 0:
        return None

    lines = content.split("\n")
    node_types = set(language.node_types)
    chunks: list[Chunk] = []

    # Accumulator for non-matching nodes (imports, comments, etc.)
    pending_start: int | None = None
    pending_end: int | None = None
    pending_lines: list[str] = []

    def flush_pending():
        """Merge pending non-matching lines into the previous chunk or start a new one."""
        nonlocal pending_start, pending_end, pending_lines
        if pending_start is None:
            return
        pending_content = "\n".join(pending_lines)
        if not pending_content.strip():
            pending_start = None
            pending_end = None
            pending_lines = []
            return
        if chunks:
            # Merge into the previous chunk
            prev = chunks[-1]
            chunks[-1] = Chunk(
                start_line=prev.start_line,
                end_line=pending_end,
                content=prev.content + "\n" + pending_content,
            )
        else:
            # No previous chunk — this becomes the start of a new chunk
            chunks.append(Chunk(
                start_line=pending_start,
                end_line=pending_end,
                content=pending_content,
            ))
        pending_start = None
        pending_end = None
        pending_lines = []

    for child in root.children:
        child_start = child.start_point[0]  # 0-indexed row
        child_end = child.end_point[0]  # 0-indexed row

        start_line = child_start + 1  # 1-indexed
        end_line = child_end + 1  # 1-indexed

        node_lines = lines[child_start : child_end + 1]
        node_content = "\n".join(node_lines)

        if child.type in node_types:
            # Flush any pending non-matching content first
            flush_pending()
            chunks.append(Chunk(
                start_line=start_line,
                end_line=end_line,
                content=node_content,
            ))
        else:
            # Accumulate non-matching nodes
            if pending_start is None:
                pending_start = start_line
            pending_end = end_line
            pending_lines.extend(node_lines)

    # Flush any trailing non-matching content
    flush_pending()

    return chunks if chunks else None


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
