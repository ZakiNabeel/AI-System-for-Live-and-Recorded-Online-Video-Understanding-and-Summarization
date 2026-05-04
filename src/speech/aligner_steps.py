"""Pure helpers for transcript cleaning and sentence alignment."""

from __future__ import annotations

import math
import re
import textwrap
from dataclasses import replace
from typing import Sequence

from .hallucination_filter import is_hallucination_candidate
from .schema import Sentence, TranscriptSegment, Word


def _segment_duration(segment: TranscriptSegment) -> float:
    return max(0.0, float(segment.end) - float(segment.start))


def _word_confidence(word: Word) -> float:
    confidence = float(word.confidence)
    if math.isnan(confidence):
        return 0.0
    return confidence


def _normalize_words(segment: TranscriptSegment) -> list[Word]:
    words = sorted(segment.words, key=lambda item: (item.start, item.end, item.text))
    normalized: list[Word] = []
    current_start = float(segment.start)
    for word in words:
        start = max(current_start, float(word.start))
        end = max(start, float(word.end))
        if end == start:
            end = start + 0.01
        normalized.append(
            replace(
                word,
                start=start,
                end=end,
                text=str(word.text).strip(),
            )
        )
        current_start = end
    return normalized


def _words_for_segment(segment: TranscriptSegment) -> list[Word]:
    words = _normalize_words(segment)
    if words:
        return words

    text = segment.text.strip()
    if not text:
        return []

    tokens = [token for token in text.split() if token]
    if not tokens:
        return []

    duration = _segment_duration(segment)
    if duration <= 0:
        duration = 0.01 * len(tokens)

    step = duration / len(tokens)
    current = float(segment.start)
    synthesized: list[Word] = []
    for index, token in enumerate(tokens):
        start = current
        end = float(segment.end) if index == len(tokens) - 1 else current + step
        if end <= start:
            end = start + 0.01
        synthesized.append(Word(start=start, end=end, text=token, confidence=math.nan))
        current = end
    return synthesized


def polish_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip())
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = cleaned.replace(" .", ".").replace(" ,", ",")
    if not cleaned:
        return cleaned

    chars = list(cleaned)
    for index, char in enumerate(chars):
        if char.isalpha():
            chars[index] = char.upper()
            break
    return "".join(chars)


def split_sentences(text: str, language: str | None = None) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []

    try:
        import pysbd  # type: ignore[import-not-found]

        segmenter = pysbd.Segmenter(language=language or "en", clean=False)
        sentences = [piece.strip() for piece in segmenter.segment(cleaned) if piece.strip()]
        if sentences:
            return sentences
    except (ImportError, LookupError, ValueError):
        pass

    sentences = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", cleaned) if piece.strip()]
    if sentences:
        return sentences
    return [cleaned]


def drop_hallucinations(segments: Sequence[TranscriptSegment]) -> list[TranscriptSegment]:
    cleaned: list[TranscriptSegment] = []
    for segment in segments:
        words = list(segment.words)
        low_confidence = not words or all(_word_confidence(word) < 0.3 for word in words)
        if is_hallucination_candidate(segment.text) and low_confidence:
            continue
        cleaned.append(
            replace(
                segment,
                text=segment.text.strip(),
                words=_normalize_words(segment),
            )
        )
    return cleaned


def _shift_segment(segment: TranscriptSegment, delta: float) -> TranscriptSegment:
    if delta == 0:
        return segment

    words = [
        replace(word, start=float(word.start) + delta, end=float(word.end) + delta)
        for word in _normalize_words(segment)
    ]
    return replace(
        segment,
        start=float(segment.start) + delta,
        end=float(segment.end) + delta,
        words=words,
    )


def fix_gaps_and_overlaps(segments: Sequence[TranscriptSegment]) -> list[TranscriptSegment]:
    fixed: list[TranscriptSegment] = []
    for segment in segments:
        current = replace(segment, text=segment.text.strip(), words=_normalize_words(segment))
        if fixed and current.start < fixed[-1].end:
            delta = float(fixed[-1].end) + 0.01 - float(current.start)
            current = _shift_segment(current, delta)
        if current.end < current.start:
            current = replace(current, end=current.start)
        fixed.append(current)
    return fixed


def _merge_segments(left: TranscriptSegment, right: TranscriptSegment) -> TranscriptSegment:
    words = sorted(
        _normalize_words(left) + _normalize_words(right),
        key=lambda item: (item.start, item.end, item.text),
    )
    text = polish_text(f"{left.text} {right.text}")
    language = left.language if left.language == right.language else left.language or right.language
    speaker = left.speaker if left.speaker == right.speaker else None
    start = min(float(left.start), float(words[0].start) if words else float(left.start))
    end = max(float(right.end), float(words[-1].end) if words else float(right.end))
    return TranscriptSegment(
        start=start,
        end=end,
        text=text,
        words=words,
        language=language,
        speaker=speaker,
    )


def merge_tiny_segments(segments: Sequence[TranscriptSegment], min_segment_sec: float) -> list[TranscriptSegment]:
    items = list(segments)
    if len(items) < 2:
        return items

    changed = True
    while changed and len(items) > 1:
        changed = False
        for index, segment in enumerate(items):
            if _segment_duration(segment) >= min_segment_sec:
                continue

            if index == 0:
                items[1] = _merge_segments(segment, items[1])
                del items[0]
                changed = True
                break
            if index == len(items) - 1:
                items[-2] = _merge_segments(items[-2], segment)
                del items[-1]
                changed = True
                break

            prev_gap = max(0.0, float(segment.start) - float(items[index - 1].end))
            next_gap = max(0.0, float(items[index + 1].start) - float(segment.end))
            if prev_gap <= next_gap:
                items[index - 1] = _merge_segments(items[index - 1], segment)
                del items[index]
            else:
                items[index + 1] = _merge_segments(segment, items[index + 1])
                del items[index]
            changed = True
            break
    return items


def _chunk_words_by_duration(words: list[Word], max_segment_sec: float) -> list[list[Word]]:
    if not words:
        return []

    chunks: list[list[Word]] = []
    start_index = 0
    while start_index < len(words):
        end_index = start_index + 1
        while end_index < len(words) and float(words[end_index].end) - float(words[start_index].start) <= max_segment_sec:
            end_index += 1
        if end_index == start_index + 1 and end_index < len(words):
            end_index += 1
        chunks.append(words[start_index:end_index])
        start_index = end_index
    return chunks


def _build_segment_from_words(
    template: TranscriptSegment,
    words: list[Word],
    text: str,
) -> TranscriptSegment:
    if words:
        start = float(words[0].start)
        end = float(words[-1].end)
    else:
        start = float(template.start)
        end = float(template.end)
    if end < start:
        end = start
    return TranscriptSegment(
        start=start,
        end=end,
        text=polish_text(text),
        words=words,
        language=template.language,
        speaker=template.speaker,
    )


def _split_segment_by_sentences(segment: TranscriptSegment, max_segment_sec: float) -> list[TranscriptSegment]:
    sentence_texts = split_sentences(segment.text, segment.language)
    words = _normalize_words(segment)

    if len(sentence_texts) <= 1:
        if words:
            chunks = _chunk_words_by_duration(words, max_segment_sec)
            return [
                _build_segment_from_words(segment, chunk, " ".join(word.text for word in chunk))
                for chunk in chunks
            ]

        text_chunks = textwrap.wrap(segment.text.strip(), width=84) or [segment.text.strip()]
        if not text_chunks:
            return [segment]
        duration = _segment_duration(segment)
        total_chars = sum(len(chunk) for chunk in text_chunks) or len(text_chunks)
        current = float(segment.start)
        result: list[TranscriptSegment] = []
        for index, chunk in enumerate(text_chunks):
            share = len(chunk) / total_chars if total_chars else 1.0 / len(text_chunks)
            end = float(segment.end) if index == len(text_chunks) - 1 else current + duration * share
            result.append(
                TranscriptSegment(
                    start=current,
                    end=end,
                    text=polish_text(chunk),
                    words=[],
                    language=segment.language,
                    speaker=segment.speaker,
                )
            )
            current = end
        return result

    if not words:
        duration = _segment_duration(segment)
        total_chars = sum(len(item) for item in sentence_texts) or len(sentence_texts)
        current = float(segment.start)
        result: list[TranscriptSegment] = []
        for index, sentence_text in enumerate(sentence_texts):
            share = len(sentence_text) / total_chars if total_chars else 1.0 / len(sentence_texts)
            end = float(segment.end) if index == len(sentence_texts) - 1 else current + duration * share
            result.append(
                TranscriptSegment(
                    start=current,
                    end=end,
                    text=polish_text(sentence_text),
                    words=[],
                    language=segment.language,
                    speaker=segment.speaker,
                )
            )
            current = end
        return result

    token_counts = [max(1, len(sentence.split())) for sentence in sentence_texts]
    remaining_words = list(words)
    result: list[TranscriptSegment] = []
    for index, sentence_text in enumerate(sentence_texts):
        if index == len(sentence_texts) - 1:
            chunk = remaining_words
        else:
            expected = token_counts[index]
            available = len(remaining_words)
            remaining_sentences = len(sentence_texts) - index - 1
            upper_bound = max(1, available - remaining_sentences)
            take = min(expected, upper_bound)
            chunk = remaining_words[:take]
            remaining_words = remaining_words[take:]
        if not chunk and remaining_words:
            chunk = [remaining_words.pop(0)]
        result.append(
            _build_segment_from_words(
                segment,
                chunk,
                sentence_text,
            )
        )
        if index != len(sentence_texts) - 1 and not remaining_words:
            break
    if remaining_words and result:
        last = result[-1]
        combined_words = last.words + remaining_words
        result[-1] = _build_segment_from_words(
            segment,
            combined_words,
            last.text,
        )
    return result


def split_overlong_segments(segments: Sequence[TranscriptSegment], max_segment_sec: float) -> list[TranscriptSegment]:
    result: list[TranscriptSegment] = []
    for segment in segments:
        if _segment_duration(segment) <= max_segment_sec:
            result.append(segment)
            continue
        result.extend(_split_segment_by_sentences(segment, max_segment_sec))
    return result


def _flatten_words_with_segments(segments: Sequence[TranscriptSegment]) -> list[tuple[int, Word]]:
    flattened: list[tuple[int, Word]] = []
    for segment_index, segment in enumerate(segments):
        for word in _normalize_words(segment):
            flattened.append((segment_index, word))
    return sorted(flattened, key=lambda item: (item[1].start, item[1].end, item[1].text))


def _polish_sentence_text(text: str) -> str:
    return polish_text(text)


def build_sentences(segments: Sequence[TranscriptSegment], language: str | None = None) -> list[Sentence]:
    if not segments:
        return []

    all_words = _flatten_words_with_segments(segments)
    full_text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
    sentence_texts = [
        _polish_sentence_text(item)
        for item in split_sentences(full_text, language)
        if item.strip()
    ]
    if not sentence_texts and full_text.strip():
        sentence_texts = [_polish_sentence_text(full_text)]

    if not all_words:
        built: list[Sentence] = []
        for segment_index, segment in enumerate(segments):
            if not segment.text.strip():
                continue
            piece_texts = split_sentences(segment.text, segment.language) or [segment.text]
            duration = _segment_duration(segment)
            total_chars = sum(len(item) for item in piece_texts) or len(piece_texts)
            current = float(segment.start)
            for index, piece in enumerate(piece_texts):
                share = len(piece) / total_chars if total_chars else 1.0 / len(piece_texts)
                end = float(segment.end) if index == len(piece_texts) - 1 else current + duration * share
                built.append(
                    Sentence(
                        start=current,
                        end=end,
                        text=_polish_sentence_text(piece),
                        word_count=len(piece.split()),
                        segment_indices=[segment_index],
                    )
                )
                current = end
        return built

    built: list[Sentence] = []
    cursor = 0
    total_words = len(all_words)
    for index, sentence_text in enumerate(sentence_texts):
        if cursor >= total_words:
            break
        if index == len(sentence_texts) - 1:
            chunk = all_words[cursor:]
        else:
            expected = max(1, len(sentence_text.split()))
            remaining_sentences = len(sentence_texts) - index - 1
            available = total_words - cursor
            take = min(expected, max(1, available - remaining_sentences))
            chunk = all_words[cursor : cursor + take]
        if not chunk:
            continue
        start = float(chunk[0][1].start)
        end = float(chunk[-1][1].end)
        segment_indices = sorted({segment_index for segment_index, _ in chunk})
        built.append(
            Sentence(
                start=start,
                end=end,
                text=sentence_text,
                word_count=len(chunk),
                segment_indices=segment_indices,
            )
        )
        cursor += len(chunk)

    if cursor < total_words and built:
        chunk = all_words[cursor:]
        last = built[-1]
        built[-1] = Sentence(
            start=last.start,
            end=float(chunk[-1][1].end),
            text=last.text,
            word_count=last.word_count + len(chunk),
            segment_indices=sorted(set(last.segment_indices + [segment_index for segment_index, _ in chunk])),
        )
    return built
