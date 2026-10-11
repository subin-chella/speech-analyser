import json
import sys
import unittest
import tempfile
import types
import wave
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

from audio_analysis import assess_pronunciation_with_azure
from audio_analysis import (
    analyze_local_audio,
    _word_emphasis_cues,
    _speaker_word_chunks,
    _write_wav_segment,
    assign_words_to_speakers,
    build_speaker_summaries,
    normalized_audio_file,
)
from storage import initialize_database, list_reports, save_report
from coaching import _parse_json_object, analyze_transcript


class AudioAnalysisTests(unittest.TestCase):
    def test_assigns_words_to_best_overlapping_speaker_turn(self):
        turns = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0},
            {"speaker": "SPEAKER_01", "start": 1.0, "end": 2.0},
        ]
        words = [
            {"word": "hello", "start": 0.1, "end": 0.5},
            {"word": "there", "start": 0.7, "end": 1.3},
        ]

        assigned = assign_words_to_speakers(words, turns)

        self.assertEqual(assigned[0]["speaker"], "SPEAKER_00")
        self.assertIsNone(assigned[1]["speaker"])

    def test_summarizes_each_speaker_and_only_within_turn_pauses(self):
        turns = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 3.0},
            {"speaker": "SPEAKER_01", "start": 3.0, "end": 4.0},
            {"speaker": "SPEAKER_02", "start": 4.0, "end": 6.0},
        ]
        words = [
            {
                "word": "hello",
                "start": 0.1,
                "end": 0.5,
                "speaker": "SPEAKER_00",
                "turn_index": 0,
            },
            {
                "word": "again",
                "start": 1.6,
                "end": 2.0,
                "speaker": "SPEAKER_00",
                "turn_index": 0,
            },
            {
                "word": "yes",
                "start": 3.2,
                "end": 3.6,
                "speaker": "SPEAKER_01",
                "turn_index": 1,
            },
        ]

        summaries = build_speaker_summaries(words, turns, pause_threshold=0.8)

        self.assertEqual([item["speaker_id"] for item in summaries], ["SPEAKER_00", "SPEAKER_01"])
        self.assertEqual(summaries[0]["within_turn_pause_count"], 1)
        self.assertEqual(summaries[0]["word_count"], 2)
        self.assertEqual(summaries[0]["pause_moments"][0]["after"], "hello")
        self.assertEqual(summaries[0]["pause_moments"][0]["before"], "again")
        self.assertEqual(summaries[1]["word_count"], 1)

    def test_word_emphasis_cues_compare_content_words_to_neighbors(self):
        class FakeSegment:
            def __init__(self, pitch, intensity):
                self.pitch = pitch
                self.intensity = intensity

            def to_pitch(self, **_kwargs):
                return types.SimpleNamespace(selected_array={"frequency": self.pitch})

            def to_intensity(self, **_kwargs):
                return types.SimpleNamespace(values=[self.intensity])

        class FakeSound:
            def get_total_duration(self):
                return 1.0

            def extract_part(self, start, _end, **_kwargs):
                if start < 0.5:
                    return FakeSegment([100.0, 100.0], [50.0, 50.0])
                return FakeSegment([120.0, 120.0], [55.0, 55.0])

        words = [
            {"word": "sales", "start": 0.1, "end": 0.4, "speaker": "SPEAKER_00", "turn_index": 0},
            {"word": "growth", "start": 0.6, "end": 0.9, "speaker": "SPEAKER_00", "turn_index": 0},
        ]
        with patch("parselmouth.Sound", return_value=FakeSound()):
            cues = _word_emphasis_cues("unused.wav", words)

        self.assertEqual(cues["SPEAKER_00"]["content_words_for_review"][0]["word"], "sales")
        self.assertEqual(cues["SPEAKER_00"]["more_emphasized_words"][0]["word"], "growth")

    def test_single_speaker_mode_constrains_pyannote_to_one_speaker(self):
        class FakeAnnotation:
            def itertracks(self, yield_label=False):
                yield (types.SimpleNamespace(start=0.0, end=1.0), None, "SPEAKER_00")

        diarization_pipeline = Mock(
            return_value=types.SimpleNamespace(
                exclusive_speaker_diarization=FakeAnnotation()
            )
        )
        transcription_model = Mock()
        transcription_model.forced_align.return_value.words = [
            types.SimpleNamespace(word="hello", start=0.1, end=0.8)
        ]

        with patch("audio_analysis._prosody_for_turns", return_value={}), patch(
            "audio_analysis._word_emphasis_cues",
            return_value={"SPEAKER_00": {"more_emphasized_words": [], "content_words_for_review": []}},
        ):
            result = analyze_local_audio(
                "meeting.wav",
                "hello",
                transcription_model,
                diarization_pipeline,
                num_speakers=1,
            )

        diarization_pipeline.assert_called_once_with("meeting.wav", num_speakers=1)
        self.assertEqual(result["speaker_count"], 1)
        self.assertEqual(result["speakers"][0]["speaker_id"], "SPEAKER_00")

    def test_cloud_assessment_chunks_split_by_gap_and_duration(self):
        words = [
            {"word": "one", "start": 0.0, "end": 0.4, "speaker": "SPEAKER_00"},
            {"word": "two", "start": 0.5, "end": 0.9, "speaker": "SPEAKER_00"},
            {"word": "three", "start": 4.0, "end": 4.4, "speaker": "SPEAKER_00"},
            {"word": "four", "start": 4.5, "end": 4.9, "speaker": "SPEAKER_01"},
        ]

        chunks = _speaker_word_chunks(words, max_duration_seconds=2.0)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0]["reference_text"], "one two")
        self.assertEqual(chunks[1]["speaker_id"], "SPEAKER_00")
        self.assertEqual(chunks[2]["speaker_id"], "SPEAKER_01")

    def test_wav_segment_header_matches_segment_frames(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as source_file:
            source_path = source_file.name
        try:
            with wave.open(source_path, "wb") as source:
                source.setnchannels(1)
                source.setsampwidth(2)
                source.setframerate(16000)
                source.writeframes(b"\x00\x00" * 16000)

            segment_path = _write_wav_segment(source_path, 0.25, 0.75)
            try:
                with wave.open(segment_path, "rb") as segment:
                    self.assertEqual(segment.getnframes(), 8000)
                    self.assertEqual(segment.getframerate(), 16000)
            finally:
                Path(segment_path).unlink(missing_ok=True)
        finally:
            Path(source_path).unlink(missing_ok=True)

    def test_normalization_failure_removes_both_temporary_files(self):
        temporary_paths = []

        def fail_conversion(arguments, **_kwargs):
            temporary_paths.extend((arguments[arguments.index("-i") + 1], arguments[-1]))
            raise RuntimeError("conversion failed")

        with patch("audio_analysis.subprocess.run", side_effect=fail_conversion):
            with self.assertRaisesRegex(RuntimeError, "conversion failed"):
                with normalized_audio_file(b"audio", "meeting.mp3"):
                    self.fail("normalization should not yield after conversion failure")

        self.assertEqual(len(temporary_paths), 2)
        self.assertTrue(all(not Path(path).exists() for path in temporary_paths))

    def test_azure_adapter_parses_scores_without_network_calls(self):
        response = {
            "NBest": [
                {
                    "PronunciationAssessment": {
                        "PronScore": 82,
                        "AccuracyScore": 78,
                        "FluencyScore": 85,
                        "ProsodyScore": 80,
                    },
                    "Words": [
                        {
                            "Word": "tested",
                            "PronunciationAssessment": {
                                "AccuracyScore": 55,
                                "ErrorType": "Mispronunciation",
                            },
                        }
                    ],
                }
            ]
        }

        class FakeSpeechConfig:
            def __init__(self, **_kwargs):
                self.speech_recognition_language = ""

        class FakeAssessmentConfig:
            def __init__(self, **_kwargs):
                pass

            def enable_prosody_assessment(self):
                pass

            def apply_to(self, _recognizer):
                pass

        class FakeRecognizer:
            def __init__(self, **_kwargs):
                pass

            def recognize_once(self):
                return types.SimpleNamespace(
                    properties=types.SimpleNamespace(
                        get=lambda _key: json.dumps(response)
                    )
                )

        speech_sdk = types.ModuleType("azure.cognitiveservices.speech")
        speech_sdk.SpeechConfig = FakeSpeechConfig
        speech_sdk.AudioConfig = lambda **_kwargs: object()
        speech_sdk.SpeechRecognizer = FakeRecognizer
        speech_sdk.PronunciationAssessmentConfig = FakeAssessmentConfig
        speech_sdk.PronunciationAssessmentGradingSystem = types.SimpleNamespace(HundredMark=1)
        speech_sdk.PronunciationAssessmentGranularity = types.SimpleNamespace(Phoneme=1)
        speech_sdk.PropertyId = types.SimpleNamespace(SpeechServiceResponse_JsonResult=1)
        azure_module = types.ModuleType("azure")
        azure_module.__path__ = []
        cognitive_module = types.ModuleType("azure.cognitiveservices")
        cognitive_module.__path__ = []
        azure_module.cognitiveservices = cognitive_module
        cognitive_module.speech = speech_sdk

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as source_file:
            source_path = source_file.name
        try:
            with wave.open(source_path, "wb") as source:
                source.setnchannels(1)
                source.setsampwidth(2)
                source.setframerate(16000)
                source.writeframes(b"\x00\x00" * 16000)

            with patch.dict(
                sys.modules,
                {
                    "azure": azure_module,
                    "azure.cognitiveservices": cognitive_module,
                    "azure.cognitiveservices.speech": speech_sdk,
                },
            ):
                result = assess_pronunciation_with_azure(
                    source_path,
                    [{"word": "tested", "start": 0.1, "end": 0.4, "speaker": "SPEAKER_00"}],
                    "fake-key",
                    "fake-region",
                )

            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["speakers"][0]["pronunciation_score"], 82.0)
            self.assertEqual(
                result["speakers"][0]["low_scoring_words"][0]["error_type"],
                "Mispronunciation",
            )
        finally:
            Path(source_path).unlink(missing_ok=True)

    def test_report_round_trip_preserves_audio_and_legacy_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "reports.sqlite3"
            initialize_database(database_path)
            audio_result = {"local": {"status": "complete", "speakers": []}}
            save_report(
                database_path,
                date(2026, 10, 10),
                "meeting.wav",
                "",
                "We discussed the launch.",
                {"word_count": 4, "audio_analysis": audio_result},
                {"summary": "Transcript coaching"},
            )
            save_report(
                database_path,
                date(2026, 10, 9),
                "legacy.txt",
                "",
                "A previous transcript.",
                {"word_count": 3},
                {"summary": "Older report"},
            )

            reports = list_reports(database_path)

            self.assertEqual(reports[0]["metrics"]["audio_analysis"], audio_result)
            self.assertNotIn("audio_analysis", reports[1]["metrics"])
            self.assertEqual(reports[1]["analysis"]["summary"], "Older report")

    def test_audio_coaching_uses_measurements_and_communication_goal(self):
        response_data = {
            "summary": "Clear contribution.",
            "scores": {"fluency": 80, "clarity": 80, "grammar": 80, "concision": 80},
            "strengths": [],
            "patterns": [],
            "practice_plan": {},
            "audio_coaching": [
                {
                    "speaker_id": "SPEAKER_00",
                    "category": "Pauses",
                    "observation": "A pause could help the main point land.",
                    "evidence": "121 words per minute; 5 within-turn pauses.",
                    "suggestion": "Pause after the key point.",
                    "practice": "Mark one pause after each key point.",
                    "success_metric": "Compare the pause placement in your next meeting.",
                }
            ],
        }
        response = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content=json.dumps(response_data))
                )
            ]
        )
        metrics = {
            "word_count": 120,
            "audio_analysis": {
                "local": {
                    "status": "complete",
                    "speakers": [
                        {
                            "speaker_id": "SPEAKER_00",
                            "speech_rate_wpm": 121.0,
                            "within_turn_pause_count": 5,
                            "pitch_spread_semitones": 1.8,
                            "turn_pitch_spread_semitones": [1.5, 1.8],
                            "pause_moments": [
                                {
                                    "seconds": 1.2,
                                    "after": "we finished the release",
                                    "before": "next we reviewed support",
                                }
                            ],
                            "word_emphasis_cues": {
                                "more_emphasized_words": [{"word": "launch"}],
                                "content_words_for_review": [{"word": "growth"}],
                            },
                        }
                    ],
                }
            },
        }

        with patch("litellm.completion", return_value=response) as completion:
            analysis = analyze_transcript(
                "The meeting starts tomorrow.",
                metrics,
                "",
                "ollama_chat/llama3.1:8b",
                "",
                api_base="http://localhost:11434",
                communication_goal="Speak clearly in status updates",
            )

        prompt = completion.call_args.kwargs["messages"][1]["content"]
        self.assertIn("121.0", prompt)
        self.assertIn("turn_pitch_spread_semitones", prompt)
        self.assertIn("we finished the release", prompt)
        self.assertIn("growth", prompt)
        self.assertIn("Speak clearly in status updates", prompt)
        self.assertEqual(analysis["audio_coaching"][0]["category"], "Pauses")

    def test_model_json_parser_accepts_fences_and_preambles(self):
        self.assertEqual(
            _parse_json_object('```json\n{"summary": "ok"}\n```'),
            {"summary": "ok"},
        )
        self.assertEqual(
            _parse_json_object('Here is the report: {"summary": "ok"}'),
            {"summary": "ok"},
        )
        with self.assertRaisesRegex(ValueError, "instead of a JSON report"):
            _parse_json_object("The model returned plain text.")


if __name__ == "__main__":
    unittest.main()