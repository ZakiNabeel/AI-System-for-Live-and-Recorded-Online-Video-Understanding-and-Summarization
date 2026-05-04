from __future__ import annotations

from pathlib import Path

from src.speech.aligner import align_transcript, main
from src.speech.schema import Transcript, TranscriptSegment, Word, load_transcript, save_transcript


def _build_transcript() -> Transcript:
    return Transcript(
        segments=[
            TranscriptSegment(
                start=0.0,
                end=4.0,
                text="hello world. second sentence.",
                words=[
                    Word(0.0, 1.0, "hello", 0.9),
                    Word(1.0, 2.0, "world", 0.9),
                    Word(2.0, 3.0, "second", 0.9),
                    Word(3.0, 4.0, "sentence", 0.9),
                ],
                language="en",
            ),
            TranscriptSegment(
                start=3.5,
                end=4.0,
                text="thanks for watching",
                words=[],
                language="en",
            ),
        ],
        language="en",
        duration_sec=4.0,
        source="local-whisper",
        audio_path=Path("data/audio/run/audio.wav"),
        raw_response={"debug": True},
    )


def test_align_transcript_writes_outputs(tmp_path: Path) -> None:
    transcript = _build_transcript()
    input_path = tmp_path / "transcript.json"
    save_transcript(transcript, input_path)

    result = align_transcript(input_path, output_dir=tmp_path / "out")

    assert result.aligned_json.exists()
    assert result.srt.exists()
    assert result.vtt.exists()

    loaded = load_transcript(result.aligned_json)
    assert loaded.sentences  # type: ignore[attr-defined]
    assert loaded.sentences[0].start <= loaded.sentences[-1].end  # type: ignore[attr-defined]


def test_align_transcript_uses_uniform_timing_without_words(tmp_path: Path) -> None:
    transcript = Transcript(
        segments=[
            TranscriptSegment(
                start=0.0,
                end=4.0,
                text="first sentence. second sentence.",
                words=[],
                language="en",
            )
        ],
        language="en",
        duration_sec=4.0,
        source="youtube-subtitles",
        audio_path=None,
        raw_response=None,
    )
    input_path = tmp_path / "transcript.json"
    save_transcript(transcript, input_path)

    result = align_transcript(input_path, output_dir=tmp_path / "out")
    loaded = load_transcript(result.aligned_json)

    assert len(loaded.sentences) == 2  # type: ignore[attr-defined]
    assert loaded.sentences[0].start == 0.0  # type: ignore[attr-defined]
    assert loaded.sentences[1].end == 4.0  # type: ignore[attr-defined]


def test_align_transcript_is_deterministic(tmp_path: Path) -> None:
    transcript = _build_transcript()
    input_path = tmp_path / "transcript.json"
    save_transcript(transcript, input_path)

    first = align_transcript(input_path, output_dir=tmp_path / "out1")
    second = align_transcript(input_path, output_dir=tmp_path / "out2")

    assert first.srt.read_text(encoding="utf-8-sig") == second.srt.read_text(encoding="utf-8-sig")
    assert first.vtt.read_text(encoding="utf-8") == second.vtt.read_text(encoding="utf-8")
    assert first.aligned_json.read_text(encoding="utf-8") == second.aligned_json.read_text(encoding="utf-8")


def test_cli_main_writes_outputs(tmp_path: Path) -> None:
    transcript = _build_transcript()
    input_path = tmp_path / "transcript.json"
    save_transcript(transcript, input_path)
    out_dir = tmp_path / "cli-out"

    code = main(["--in", str(input_path), "--out-dir", str(out_dir)])

    assert code == 0
    assert (out_dir / "transcript.aligned.json").exists()
    assert (out_dir / "transcript.srt").exists()
    assert (out_dir / "transcript.vtt").exists()
