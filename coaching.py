import json
import html
import re
from typing import Any


FILLER_PATTERNS = {
    "um": r"\bum\b",
    "uh": r"\buh\b",
    "erm": r"\berm\b",
    "hmm": r"\bhmm\b",
    "you know": r"\byou know\b",
    "I mean": r"\bi mean\b",
    "sort of": r"\bsort of\b",
    "kind of": r"\bkind of\b",
    "basically": r"\bbasically\b",
    "literally": r"\bliterally\b",
}


def _parse_timestamp(value: str) -> float | None:
    match = re.fullmatch(r"(?:(\d+):)?(\d{2}):(\d{2})(?:[.,](\d{1,3}))?", value)
    if not match:
        return None
    hours, minutes, seconds, milliseconds = match.groups()
    return (
        int(hours or 0) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int((milliseconds or "0").ljust(3, "0")) / 1000
    )


def _transcript_turns(transcript: str) -> list[dict[str, Any]]:
    turns = []
    timestamp_line = re.compile(
        r"^\s*((?:\d+:)?\d{2}:\d{2}[.,]\d{1,3})\s+-->\s+"
        r"((?:\d+:)?\d{2}:\d{2}[.,]\d{1,3})(?:\s+.*)?\s*$"
    )
    voice_tag = re.compile(r"<v(?:\.[^ >]+)?\s+([^>]+)>(.*?)</v>", re.IGNORECASE | re.DOTALL)
    speaker_prefix = re.compile(r"^\s*([^:\n]{1,100}):\s*(.+?)\s*$")

    for block in re.split(r"\n\s*\n", transcript.strip()):
        lines = block.splitlines()
        timing_index = next(
            (index for index, line in enumerate(lines) if timestamp_line.match(line)),
            None,
        )
        start = end = None
        content_lines = lines
        if timing_index is not None:
            timing_match = timestamp_line.match(lines[timing_index])
            start = _parse_timestamp(timing_match.group(1))
            end = _parse_timestamp(timing_match.group(2))
            content_lines = lines[timing_index + 1 :]

        content = "\n".join(content_lines)
        voice_matches = list(voice_tag.finditer(content))
        if voice_matches:
            for match in voice_matches:
                text = html.unescape(re.sub(r"<[^>]+>", " ", match.group(2))).strip()
                if text:
                    turns.append(
                        {"speaker": match.group(1).strip(), "text": text, "start": start, "end": end}
                    )
            continue

        clean_content = html.unescape(re.sub(r"<[^>]+>", " ", content))
        for line in clean_content.splitlines():
            line = line.strip()
            if not line:
                continue
            prefix_match = speaker_prefix.match(line)
            if prefix_match:
                turns.append(
                    {
                        "speaker": prefix_match.group(1).strip(),
                        "text": prefix_match.group(2).strip(),
                        "start": start,
                        "end": end,
                    }
                )
            elif timing_index is not None:
                turns.append({"speaker": None, "text": line, "start": start, "end": end})

    return turns


def _speaker_tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.casefold()))


def _matching_speaker(turns: list[dict[str, Any]], target_label: str) -> str | None:
    target_tokens = _speaker_tokens(target_label)
    speakers = list(dict.fromkeys(turn["speaker"] for turn in turns if turn["speaker"]))
    matches = [
        speaker
        for speaker in speakers
        if target_tokens
        and (
            target_tokens.issubset(_speaker_tokens(speaker))
            or _speaker_tokens(speaker).issubset(target_tokens)
        )
    ]
    return matches[0] if len(matches) == 1 else None


def _pause_gaps(
    turns: list[dict[str, Any]],
    selected_speaker: str | None = None,
) -> list[float]:
    timed_turns = [
        turn
        for turn in turns
        if isinstance(turn.get("start"), (int, float))
        and isinstance(turn.get("end"), (int, float))
    ]
    if selected_speaker:
        selected_turns = sorted(
            (turn for turn in timed_turns if turn["speaker"] == selected_speaker),
            key=lambda turn: turn["start"],
        )
        other_intervals = sorted(
            (float(turn["start"]), float(turn["end"]))
            for turn in timed_turns
            if turn["speaker"] != selected_speaker
        )
        gaps = []
        for current, following in zip(selected_turns, selected_turns[1:]):
            cursor = float(current["end"])
            next_start = float(following["start"])
            if next_start <= cursor:
                continue
            for other_start, other_end in other_intervals:
                if other_end <= cursor or other_start >= next_start:
                    continue
                if other_start > cursor:
                    gaps.append(other_start - cursor)
                cursor = max(cursor, other_end)
            if next_start > cursor:
                gaps.append(next_start - cursor)
        return gaps

    intervals = sorted((float(turn["start"]), float(turn["end"])) for turn in timed_turns)
    gaps = []
    if intervals:
        interval_end = intervals[0][1]
        for start, end in intervals[1:]:
            if start > interval_end:
                gaps.append(start - interval_end)
            interval_end = max(interval_end, end)
    return gaps


def speech_metrics(
    transcript: str,
    word_timestamps: list[dict[str, Any]],
    duration_seconds: float,
    pause_threshold: float,
    target_label: str = "",
) -> dict[str, Any]:
    turns = _transcript_turns(transcript)
    selected_speaker = None
    metrics_available = True
    metrics_unavailable_reason = ""
    if target_label:
        selected_speaker = _matching_speaker(turns, target_label)
        if selected_speaker is None:
            metrics_available = False
            metrics_unavailable_reason = (
                "No unambiguous speaker label matched the requested participant."
            )
            selected_turns = []
        else:
            selected_turns = [turn for turn in turns if turn["speaker"] == selected_speaker]
        text_segments = [turn["text"] for turn in selected_turns]
    elif turns:
        selected_turns = turns
        text_segments = [turn["text"] for turn in turns]
    else:
        selected_turns = []
        text_segments = [transcript]

    if target_label and not any(turn["speaker"] for turn in turns):
        metrics_unavailable_reason = "The transcript has no speaker labels to isolate the participant."
    words = [
        word
        for segment in text_segments
        for word in re.findall(r"\b[\w']+\b", segment, flags=re.UNICODE)
    ]
    fillers_by_type = {
        filler: sum(len(re.findall(pattern, segment, flags=re.IGNORECASE)) for segment in text_segments)
        for filler, pattern in FILLER_PATTERNS.items()
    }
    filler_count = sum(fillers_by_type.values())
    repeated_words = sum(
        len(
            re.findall(
                r"\b((?:[\w']+\s+){0,2}[\w']+)[,;:!?]?\s+\1\b",
                segment,
                flags=re.IGNORECASE,
            )
        )
        for segment in text_segments
    )

    valid_words = [
        item
        for item in word_timestamps
        if isinstance(item.get("start"), (int, float))
        and isinstance(item.get("end"), (int, float))
    ]
    timed_turns = [
        turn
        for turn in turns
        if isinstance(turn.get("start"), (int, float))
        and isinstance(turn.get("end"), (int, float))
    ]
    selected_timed_turns = [
        turn for turn in timed_turns if turn["speaker"] == selected_speaker
    ]
    if timed_turns and (not target_label or selected_speaker):
        safe_duration = max(
            max(float(turn["end"]) for turn in timed_turns)
            - min(float(turn["start"]) for turn in timed_turns),
            0.0,
        )
        gaps = _pause_gaps(turns, selected_speaker if target_label else None)
    elif valid_words and not target_label:
        gaps = [
            float(following["start"]) - float(current["end"])
            for current, following in zip(valid_words, valid_words[1:])
            if float(following["start"]) - float(current["end"]) >= pause_threshold
        ]
        safe_duration = max(float(duration_seconds), 0.0)
    else:
        gaps = []
        safe_duration = max(float(duration_seconds), 0.0) if not target_label else 0.0

    gaps = [gap for gap in gaps if gap >= pause_threshold]
    timing_available = (
        bool(selected_timed_turns)
        if target_label
        else bool(timed_turns) or bool(valid_words and safe_duration > 0)
    )
    return {
        "word_count": len(words),
        "filler_count": filler_count,
        "fillers_by_type": fillers_by_type,
        "filler_per_1000": round(filler_count * 1000 / max(len(words), 1), 1),
        "repeated_words": repeated_words,
        "long_pause_count": len(gaps),
        "longest_pause_seconds": round(max(gaps, default=0.0), 1),
        "long_pauses_per_minute": round(len(gaps) * 60 / max(safe_duration, 1.0), 1),
        "duration_seconds": round(safe_duration, 1),
        "timing_available": timing_available,
        "metrics_available": metrics_available,
        "speaker_label": selected_speaker,
        "metrics_unavailable_reason": metrics_unavailable_reason,
    }


def analyze_transcript(
    transcript: str,
    metrics: dict[str, Any],
    target_label: str,
    model_name: str,
    api_key: str,
    api_base: str | None = None,
) -> dict[str, Any]:
    from litellm import completion

    scope = (
        f"Evaluate only the participant identified by {target_label!r}. Match speaker labels flexibly: ignore capitalization and punctuation, recognize first/last-name order changes such as 'Prakash Mani' and 'Mani, Prakash', and accept a shortened first or last name only when it identifies one speaker unambiguously. Do not combine different people with similar names. If the identity or speaker turns are ambiguous or absent, say individual attribution is uncertain rather than guessing. Use only this participant's labeled turns for individual observations. "
        + (
            "The supplied objective metrics have been calculated from this participant's turns; use them as speaker-specific values."
            if metrics.get("metrics_available", True)
            else "Speaker-specific metric extraction failed. Do not use the supplied numeric metric values; explain that the transcript's speaker labels did not allow reliable isolation."
        )
        if target_label
        else "Analyze the whole transcript or recording, considering all participants together. The supplied objective metrics describe the whole recording; do not attribute observations or metrics to an individual speaker."
    )
    timing_guidance = (
        "No word timing is available, so do not make claims about measured pauses or speaking rate."
        if not metrics.get("timing_available", True)
        else (
            "The pause metrics are calculated for the selected participant's turns only."
            if target_label
            else "Timing metrics describe the full recording."
        )
    )
    prompt = f"""You are a practical communication coach reviewing a meeting transcript.
Be respectful and specific. Treat spoken language as informal; flag only clear grammatical issues, not acceptable conversational style.
{scope}
Use the objective metrics as evidence, and distinguish measured facts from interpretations. {timing_guidance} Give no medical or personality judgments.
Use fillers_by_type to name the most frequent filler only when its count is nonzero. Suggest a practical replacement, such as pausing silently while thinking. Do not claim a filler is frequent when counts are zero, and do not invent measurements.
Return only a JSON object with this exact shape:
{{
  "summary": "short overall assessment",
  "scores": {{"fluency": 0, "clarity": 0, "grammar": 0, "concision": 0}},
  "strengths": ["specific strength"],
  "patterns": [{{"category": "Filler words, Pauses, Repetition, Grammar, or Clarity", "observation": "specific observation", "evidence": "short quote or measured evidence", "suggestion": "one actionable adjustment", "priority": "High, Medium, or Low"}}],
  "practice_plan": {{"focus": "one behavior", "exercise": "a short practice exercise", "success_metric": "a measurable target for the next meeting"}}
}}
Scores must be integers from 0 to 100. Include no more than four patterns and one practice focus.

Objective speech metrics:
{json.dumps(metrics, ensure_ascii=True)}

Transcript:
{transcript}
"""
    arguments: dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "Return valid JSON only, following the user's requested schema."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    if api_key:
        arguments["api_key"] = api_key
    if api_base:
        arguments["api_base"] = api_base

    try:
        response = completion(**arguments)
    except Exception as error:
        if api_base:
            model_to_pull = model_name.removeprefix("ollama_chat/").removeprefix("ollama/")
            if "not found" in str(error).lower():
                raise RuntimeError(
                    f"The Ollama model '{model_to_pull}' is not downloaded. In another Terminal, run: ollama pull {model_to_pull}"
                ) from error
            raise RuntimeError(
                f"Could not reach the local LLM at {api_base}. Make sure Ollama is running. Details: {error}"
            ) from error
        raise
    content = response.choices[0].message.content
    if not content:
        raise ValueError("The selected model returned an empty coaching report.")
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
    analysis = json.loads(cleaned)
    required = {"summary", "scores", "strengths", "patterns", "practice_plan"}
    if not required.issubset(analysis):
        raise ValueError("The selected model did not return all required report fields.")
    for score_name in ("fluency", "clarity", "grammar", "concision"):
        analysis["scores"][score_name] = max(0, min(100, int(analysis["scores"].get(score_name, 0))))
    return analysis