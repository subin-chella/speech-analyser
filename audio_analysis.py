from __future__ import annotations

import math
import json
import statistics
import subprocess
import tempfile
import wave
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any


_FUNCTION_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her",
    "here", "him", "his", "i", "if", "in", "into", "is", "it", "its", "may",
    "me", "might", "my", "of", "on", "or", "our", "she", "should", "so",
    "that", "the", "their", "them", "there", "they", "this", "to", "us", "was",
    "we", "were", "what", "when", "which", "who", "will", "with", "would", "you",
    "your",
}


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


@contextmanager
def normalized_audio_file(audio_bytes: bytes, file_name: str):
    input_path = None
    output_path = None
    try:
        suffix = Path(file_name).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as source:
            source.write(audio_bytes)
            input_path = source.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as converted:
            output_path = converted.name
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                input_path,
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                output_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        yield output_path
    finally:
        if input_path:
            Path(input_path).unlink(missing_ok=True)
        if output_path:
            Path(output_path).unlink(missing_ok=True)


def assign_words_to_speakers(
    words: Sequence[Any], turns: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    assigned = []
    for word in words:
        start = _value(word, "start")
        end = _value(word, "end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        duration = float(end) - float(start)
        if duration <= 0:
            continue

        overlaps = []
        for turn_index, turn in enumerate(turns):
            overlap = min(float(end), turn["end"]) - max(float(start), turn["start"])
            if overlap > 0:
                overlaps.append((overlap, turn_index, turn["speaker"]))
        overlaps.sort(reverse=True)
        speaker = None
        turn_index = None
        coverage = 0.0
        if overlaps:
            best_overlap, best_index, best_speaker = overlaps[0]
            is_tied = len(overlaps) > 1 and math.isclose(best_overlap, overlaps[1][0])
            coverage = min(best_overlap / duration, 1.0)
            if not is_tied and coverage >= 0.5:
                speaker = best_speaker
                turn_index = best_index

        assigned.append(
            {
                "word": str(_value(word, "word", "")).strip(),
                "start": float(start),
                "end": float(end),
                "speaker": speaker,
                "turn_index": turn_index,
                "assignment_confidence": round(coverage, 3),
            }
        )
    return assigned


def build_speaker_summaries(
    words: Sequence[Mapping[str, Any]],
    turns: Sequence[Mapping[str, Any]],
    pause_threshold: float = 0.8,
) -> list[dict[str, Any]]:
    words_by_turn: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    words_by_speaker: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for word in words:
        speaker = word.get("speaker")
        turn_index = word.get("turn_index")
        if speaker and isinstance(turn_index, int):
            words_by_turn[turn_index].append(word)
            words_by_speaker[speaker].append(word)

    turn_indexes_by_speaker: dict[str, list[int]] = defaultdict(list)
    for index, turn in enumerate(turns):
        turn_indexes_by_speaker[turn["speaker"]].append(index)

    summaries = []
    for speaker, turn_indexes in turn_indexes_by_speaker.items():
        matched_words = words_by_speaker.get(speaker, [])
        if not matched_words:
            continue
        speech_seconds = sum(
            max(float(turns[index]["end"]) - float(turns[index]["start"]), 0.0)
            for index in turn_indexes
        )
        pauses = []
        for index in turn_indexes:
            turn_words = sorted(words_by_turn.get(index, []), key=lambda word: word["start"])
            for word_index, (current, following) in enumerate(zip(turn_words, turn_words[1:])):
                gap = float(following["start"]) - float(current["end"])
                if gap >= pause_threshold:
                    pauses.append(
                        {
                            "seconds": round(gap, 2),
                            "after": " ".join(
                                str(item.get("word", "")).strip()
                                for item in turn_words[max(0, word_index - 2) : word_index + 1]
                            ),
                            "before": " ".join(
                                str(item.get("word", "")).strip()
                                for item in turn_words[word_index + 1 : word_index + 4]
                            ),
                        }
                    )
        longest_pause = max((pause["seconds"] for pause in pauses), default=0.0)
        summaries.append(
            {
                "speaker_id": speaker,
                "turn_count": len(turn_indexes),
                "speech_seconds": round(speech_seconds, 2),
                "word_count": len(matched_words),
                "speech_rate_wpm": round(len(matched_words) * 60 / speech_seconds, 1)
                if speech_seconds > 0
                else None,
                "within_turn_pause_count": len(pauses),
                "longest_within_turn_pause_seconds": longest_pause,
                "pause_moments": sorted(
                    pauses,
                    key=lambda pause: pause["seconds"],
                    reverse=True,
                )[:6],
                "word_assignment_coverage": round(
                    len(matched_words) / max(len(words), 1), 3
                ),
            }
        )
    return summaries


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _prosody_for_turns(audio_path: str, turns: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    import parselmouth

    sound = parselmouth.Sound(audio_path)
    pitches = []
    intensities = []
    turn_pitch_spreads = []
    for turn in turns:
        if float(turn["end"]) - float(turn["start"]) < 0.2:
            continue
        segment = sound.extract_part(
            float(turn["start"]),
            float(turn["end"]),
            preserve_times=False,
        )
        pitch = segment.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
        turn_pitches = [
            float(value)
            for value in pitch.selected_array["frequency"]
            if math.isfinite(float(value)) and float(value) > 0
        ]
        pitches.extend(turn_pitches)
        turn_pitch_p10 = _percentile(turn_pitches, 10)
        turn_pitch_p90 = _percentile(turn_pitches, 90)
        if turn_pitch_p10 and turn_pitch_p90 and turn_pitch_p10 > 0:
            turn_pitch_spreads.append(12 * math.log2(turn_pitch_p90 / turn_pitch_p10))
        intensity = segment.to_intensity(minimum_pitch=75, time_step=0.01)
        intensities.extend(
            float(value)
            for value in intensity.values[0]
            if math.isfinite(float(value))
        )

    pitch_p10 = _percentile(pitches, 10)
    pitch_p90 = _percentile(pitches, 90)
    pitch_spread_semitones = (
        12 * math.log2(pitch_p90 / pitch_p10)
        if pitch_p10 is not None and pitch_p90 is not None and pitch_p10 > 0
        else None
    )
    intensity_p10 = _percentile(intensities, 10)
    intensity_p90 = _percentile(intensities, 90)
    return {
        "pitch_median_hz": round(statistics.median(pitches), 1) if pitches else None,
        "pitch_p10_hz": round(pitch_p10, 1) if pitch_p10 is not None else None,
        "pitch_p90_hz": round(pitch_p90, 1) if pitch_p90 is not None else None,
        "pitch_spread_hz": round(pitch_p90 - pitch_p10, 1)
        if pitch_p10 is not None and pitch_p90 is not None
        else None,
        "pitch_spread_semitones": round(pitch_spread_semitones, 2)
        if pitch_spread_semitones is not None
        else None,
        "turn_pitch_spread_semitones": [round(value, 2) for value in turn_pitch_spreads],
        "intensity_p10_db": round(intensity_p10, 1) if intensity_p10 is not None else None,
        "intensity_p90_db": round(intensity_p90, 1) if intensity_p90 is not None else None,
        "intensity_spread_db": round(intensity_p90 - intensity_p10, 1)
        if intensity_p10 is not None and intensity_p90 is not None
        else None,
    }


def _word_emphasis_cues(
    audio_path: str,
    words: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    import parselmouth

    sound = parselmouth.Sound(audio_path)
    words_by_turn: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for word in words:
        speaker = word.get("speaker")
        turn_index = word.get("turn_index")
        if speaker and isinstance(turn_index, int) and word.get("word"):
            if float(word["end"]) - float(word["start"]) >= 0.14:
                words_by_turn[(str(speaker), turn_index)].append(word)

    cues_by_speaker: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for (speaker, _), turn_words in words_by_turn.items():
        ordered_words = sorted(turn_words, key=lambda word: float(word["start"]))
        measured_words = []
        for word in ordered_words:
            start = max(float(word["start"]) - 0.025, 0.0)
            end = min(float(word["end"]) + 0.025, sound.get_total_duration())
            if end - start < 0.1:
                continue
            segment = sound.extract_part(start, end, preserve_times=False)
            pitch = segment.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
            pitch_values = [
                float(value)
                for value in pitch.selected_array["frequency"]
                if math.isfinite(float(value)) and float(value) > 0
            ]
            intensity = segment.to_intensity(minimum_pitch=75, time_step=0.01)
            intensity_values = [
                float(value)
                for value in intensity.values[0]
                if math.isfinite(float(value))
            ]
            measured_words.append(
                {
                    "word": str(word["word"]).strip(),
                    "pitch_median_hz": statistics.median(pitch_values) if pitch_values else None,
                    "intensity_median_db": statistics.median(intensity_values)
                    if intensity_values
                    else None,
                }
            )

        emphasized = []
        underemphasized = []
        for index, word in enumerate(measured_words):
            neighbors = measured_words[max(0, index - 2) : index] + measured_words[index + 1 : index + 3]
            neighbor_pitch = [
                neighbor["pitch_median_hz"]
                for neighbor in neighbors
                if neighbor["pitch_median_hz"] is not None
            ]
            neighbor_intensity = [
                neighbor["intensity_median_db"]
                for neighbor in neighbors
                if neighbor["intensity_median_db"] is not None
            ]
            if not neighbor_pitch and not neighbor_intensity:
                continue
            pitch_delta_semitones = None
            if word["pitch_median_hz"] and neighbor_pitch:
                pitch_delta_semitones = 12 * math.log2(
                    word["pitch_median_hz"] / statistics.median(neighbor_pitch)
                )
            intensity_delta_db = None
            if word["intensity_median_db"] is not None and neighbor_intensity:
                intensity_delta_db = word["intensity_median_db"] - statistics.median(
                    neighbor_intensity
                )

            cue = {
                "word": word["word"],
                "pitch_change_semitones": round(pitch_delta_semitones, 2)
                if pitch_delta_semitones is not None
                else None,
                "intensity_change_db": round(intensity_delta_db, 1)
                if intensity_delta_db is not None
                else None,
            }
            has_emphasis = (
                (pitch_delta_semitones is not None and pitch_delta_semitones >= 0.8)
                or (intensity_delta_db is not None and intensity_delta_db >= 2.5)
            )
            if has_emphasis:
                emphasized.append(cue)

            normalized_word = "".join(character for character in word["word"].casefold() if character.isalnum())
            is_content_word = len(normalized_word) >= 3 and normalized_word not in _FUNCTION_WORDS
            has_low_prominence = (
                is_content_word
                and intensity_delta_db is not None
                and intensity_delta_db <= -2.5
                and (pitch_delta_semitones is None or pitch_delta_semitones <= 0.3)
            )
            if has_low_prominence:
                underemphasized.append(cue)

        speaker_cues = cues_by_speaker.setdefault(
            speaker,
            {"more_emphasized_words": [], "content_words_for_review": []},
        )
        speaker_cues["more_emphasized_words"].extend(emphasized)
        speaker_cues["content_words_for_review"].extend(underemphasized)

    for cues in cues_by_speaker.values():
        cues["more_emphasized_words"] = sorted(
            cues["more_emphasized_words"],
            key=lambda cue: max(
                cue["intensity_change_db"] or 0,
                (cue["pitch_change_semitones"] or 0) * 2,
            ),
            reverse=True,
        )[:6]
        cues["content_words_for_review"] = sorted(
            cues["content_words_for_review"],
            key=lambda cue: cue["intensity_change_db"] or 0,
        )[:6]
    return cues_by_speaker


def _speaker_turns(diarization: Any) -> list[dict[str, Any]]:
    annotation = getattr(diarization, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = getattr(diarization, "speaker_diarization", None)
    if annotation is None and hasattr(diarization, "itertracks"):
        annotation = diarization
    if annotation is None or not hasattr(annotation, "itertracks"):
        raise ValueError("The diarization pipeline returned no speaker turns.")

    turns = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        start, end = float(segment.start), float(segment.end)
        if end > start:
            turns.append({"speaker": str(speaker), "start": start, "end": end})
    return sorted(turns, key=lambda turn: (turn["start"], turn["end"]))


def analyze_local_audio(
    audio_path: str,
    transcript: str,
    transcription_model: Any,
    diarization_pipeline: Any,
    pause_threshold: float = 0.8,
    num_speakers: int | None = None,
) -> dict[str, Any]:
    if not transcript.strip():
        raise ValueError("A non-empty corrected transcript is required for audio alignment.")

    diarization = (
        diarization_pipeline(audio_path, num_speakers=num_speakers)
        if num_speakers is not None
        else diarization_pipeline(audio_path)
    )
    turns = _speaker_turns(diarization)
    if not turns:
        return {"status": "unavailable", "reason": "No speech turns were detected.", "speakers": []}

    alignment = transcription_model.forced_align(
        audio_path,
        transcript,
        language="en",
        mode="verbatim",
    )
    aligned_words = list(getattr(alignment, "words", None) or [])
    assigned_words = assign_words_to_speakers(aligned_words, turns)
    summaries = build_speaker_summaries(assigned_words, turns, pause_threshold)
    emphasis_cues = _word_emphasis_cues(audio_path, assigned_words)
    for summary in summaries:
        speaker_turns = [turn for turn in turns if turn["speaker"] == summary["speaker_id"]]
        summary.update(_prosody_for_turns(audio_path, speaker_turns))
        summary["word_emphasis_cues"] = emphasis_cues.get(
            summary["speaker_id"],
            {"more_emphasized_words": [], "content_words_for_review": []},
        )

    return {
        "status": "complete",
        "backend": "local",
        "language": "en",
        "aligned_word_count": len(aligned_words),
        "assigned_word_count": sum(word["speaker"] is not None for word in assigned_words),
        "speaker_count": len(summaries),
        "speakers": summaries,
        "notes": [
            "Pitch and intensity are recording-relative acoustic measurements, not quality grades.",
            "Speaker labels are anonymous and diarization can be inaccurate during overlap.",
            "Speaker IDs without confidently aligned words are omitted from the report.",
            "Local pronunciation scoring is not available; word timing is used for pacing and pause measurements.",
        ],
        "_assigned_words": assigned_words,
    }


def _speaker_word_chunks(
    words: Sequence[Mapping[str, Any]],
    max_duration_seconds: float = 24.0,
    max_gap_seconds: float = 1.5,
) -> list[dict[str, Any]]:
    grouped_words: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for word in words:
        if word.get("speaker") and word.get("word"):
            grouped_words[str(word["speaker"])].append(word)

    chunks = []
    for speaker, speaker_words in grouped_words.items():
        ordered = sorted(speaker_words, key=lambda word: float(word["start"]))
        current: list[Mapping[str, Any]] = []
        for word in ordered:
            if current:
                elapsed = float(word["end"]) - float(current[0]["start"])
                gap = float(word["start"]) - float(current[-1]["end"])
                if gap > max_gap_seconds or elapsed > max_duration_seconds:
                    chunks.append(_make_word_chunk(speaker, current))
                    current = []
            current.append(word)
        if current:
            chunks.append(_make_word_chunk(speaker, current))
    return chunks


def _make_word_chunk(speaker: str, words: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "speaker_id": speaker,
        "start": float(words[0]["start"]),
        "end": float(words[-1]["end"]),
        "reference_text": " ".join(str(word["word"]).strip() for word in words),
        "word_count": len(words),
    }


def _write_wav_segment(audio_path: str, start: float, end: float) -> str:
    with wave.open(audio_path, "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError("Pronunciation assessment requires mono 16-bit PCM WAV audio.")
        sample_rate = source.getframerate()
        start_frame = max(int(start * sample_rate), 0)
        end_frame = min(int(end * sample_rate), source.getnframes())
        if end_frame <= start_frame:
            raise ValueError("The aligned speech segment has no audio frames.")
        source.setpos(start_frame)
        frames = source.readframes(end_frame - start_frame)
        parameters = source.getparams()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        segment_path = temp_audio.name
    with wave.open(segment_path, "wb") as segment:
        segment.setparams(
            (
                parameters.nchannels,
                parameters.sampwidth,
                parameters.framerate,
                0,
                parameters.comptype,
                parameters.compname,
            )
        )
        segment.writeframes(frames)
    return segment_path


def _assessment_score(payload: Mapping[str, Any], name: str) -> float | None:
    value = payload.get(name)
    return float(value) if isinstance(value, (int, float)) else None


def _mean_score(items: Sequence[Mapping[str, Any]], name: str) -> float | None:
    values = [score for item in items if (score := _assessment_score(item, name)) is not None]
    return round(statistics.mean(values), 1) if values else None


def assess_pronunciation_with_azure(
    audio_path: str,
    assigned_words: Sequence[Mapping[str, Any]],
    subscription_key: str,
    service_region: str,
    locale: str = "en-US",
) -> dict[str, Any]:
    if locale != "en-US":
        raise ValueError("Phoneme names and prosody assessment are currently enabled for en-US only.")
    if not subscription_key.strip() or not service_region.strip():
        raise ValueError("Azure Speech credentials and region are required for cloud pronunciation scoring.")

    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError as error:
        raise RuntimeError(
            "Azure Speech SDK is not installed. Install azure-cognitiveservices-speech to enable cloud scoring."
        ) from error

    chunks = _speaker_word_chunks(assigned_words)
    if not chunks:
        return {
            "status": "unavailable",
            "backend": "azure_speech",
            "reason": "No confidently speaker-attributed aligned words were available.",
            "speakers": [],
        }

    speech_config = speechsdk.SpeechConfig(
        subscription=subscription_key.strip(),
        region=service_region.strip(),
    )
    speech_config.speech_recognition_language = locale
    scored_chunks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failed_chunks = 0
    warnings = []
    for chunk in chunks:
        segment_path = None
        try:
            segment_path = _write_wav_segment(audio_path, chunk["start"], chunk["end"])
            audio_config = speechsdk.AudioConfig(filename=segment_path)
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_config,
            )
            assessment = speechsdk.PronunciationAssessmentConfig(
                reference_text=chunk["reference_text"],
                grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
                granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
                enable_miscue=False,
            )
            assessment.enable_prosody_assessment()
            assessment.apply_to(recognizer)
            result = recognizer.recognize_once()
            result_json = result.properties.get(
                speechsdk.PropertyId.SpeechServiceResponse_JsonResult
            )
            if not result_json:
                failed_chunks += 1
                warnings.append(f"No score returned for {chunk['speaker_id']} audio chunk.")
                continue

            response = json.loads(result_json)
            best_result = (response.get("NBest") or [{}])[0]
            scores = best_result.get("PronunciationAssessment", {})
            if not scores:
                failed_chunks += 1
                warnings.append(f"No pronunciation scores returned for {chunk['speaker_id']} audio chunk.")
                continue
            scored_words = []
            for word in best_result.get("Words", []):
                word_scores = word.get("PronunciationAssessment", {})
                accuracy = _assessment_score(word_scores, "AccuracyScore")
                error_type = word_scores.get("ErrorType")
                if accuracy is not None and (accuracy < 80 or error_type not in (None, "None")):
                    scored_words.append(
                        {
                            "word": word.get("Word", ""),
                            "accuracy_score": accuracy,
                            "error_type": error_type,
                        }
                    )
            scored_chunks[chunk["speaker_id"]].append(
                {
                    "pronunciation_score": _assessment_score(scores, "PronScore"),
                    "accuracy_score": _assessment_score(scores, "AccuracyScore"),
                    "fluency_score": _assessment_score(scores, "FluencyScore"),
                    "prosody_score": _assessment_score(scores, "ProsodyScore"),
                    "low_scoring_words": scored_words,
                    "word_count": chunk["word_count"],
                }
            )
        except Exception as error:
            failed_chunks += 1
            warnings.append(f"Scoring failed for {chunk['speaker_id']} audio chunk: {error}")
        finally:
            if segment_path:
                Path(segment_path).unlink(missing_ok=True)

    speakers = []
    for speaker_id, entries in scored_chunks.items():
        speakers.append(
            {
                "speaker_id": speaker_id,
                "pronunciation_score": _mean_score(entries, "pronunciation_score"),
                "accuracy_score": _mean_score(entries, "accuracy_score"),
                "fluency_score": _mean_score(entries, "fluency_score"),
                "prosody_score": _mean_score(entries, "prosody_score"),
                "low_scoring_words": [
                    word
                    for entry in entries
                    for word in entry["low_scoring_words"]
                ][:30],
                "assessed_chunks": len(entries),
            }
        )

    return {
        "status": "partial" if failed_chunks else "complete",
        "backend": "azure_speech",
        "locale": locale,
        "speaker_count": len(speakers),
        "speakers": speakers,
        "failed_chunk_count": failed_chunks,
        "warnings": warnings,
        "notes": [
            "Scores are Azure Speech assessment outputs, not universal or accent-neutral measures.",
            "Audio clips are sent to Azure Speech for scoring.",
            "Meeting speech can be less suitable for reference-text scoring than read speech.",
        ],
    }