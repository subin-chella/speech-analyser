import json
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


def speech_metrics(
    transcript: str,
    word_timestamps: list[dict[str, Any]],
    duration_seconds: float,
    pause_threshold: float,
) -> dict[str, Any]:
    words = re.findall(r"\b[\w']+\b", transcript, flags=re.UNICODE)
    fillers_by_type = {
        filler: len(re.findall(pattern, transcript, flags=re.IGNORECASE))
        for filler, pattern in FILLER_PATTERNS.items()
    }
    filler_count = sum(fillers_by_type.values())
    repeated_words = len(
        re.findall(
            r"\b((?:[\w']+\s+){0,2}[\w']+)[,;:!?]?\s+\1\b",
            transcript,
            flags=re.IGNORECASE,
        )
    )

    gaps: list[float] = []
    valid_words = [
        item
        for item in word_timestamps
        if isinstance(item.get("start"), (int, float))
        and isinstance(item.get("end"), (int, float))
    ]
    for current, following in zip(valid_words, valid_words[1:]):
        gap = float(following["start"]) - float(current["end"])
        if gap >= pause_threshold:
            gaps.append(gap)

    safe_duration = max(float(duration_seconds), 0.0)
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
        f"Evaluate only the participant labeled {target_label!r}. "
        "The transcript may include other speakers; do not attribute their words or pauses to this participant. "
        "If no lines have speaker labels, state that individual attribution is not possible and assess the conversation cautiously. "
        "The supplied objective metrics describe the whole recording, not a speaker-specific subset; never attribute them to this participant."
        if target_label
        else "Assess the conversation as a whole and do not claim that any observation belongs to a specific participant. "
        "The supplied objective metrics describe the whole recording."
    )
    prompt = f"""You are a practical communication coach reviewing a meeting transcript.
Be respectful and specific. Treat spoken language as informal; flag only clear grammatical issues, not acceptable conversational style.
{scope}
Use the objective metrics as evidence, and distinguish measured facts from interpretations. Give no medical or personality judgments.
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