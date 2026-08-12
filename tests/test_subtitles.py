"""
Tests for the alignment → subtitle path.

The `alignment` object (one timestamp per script character) is the spine of the whole system:
scene cuts, subtitle timing and music duration are all derived from it. These tests pin down the
two transforms that turn it into what the viewer actually sees.

    python -m pytest tests/ -q
"""

import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.subtitles import (
    SUB_WINDOW, _fmt, _esc, build_ass_rolling, words_from_alignment,
)


def linear_alignment(text, seconds_per_char=0.05):
    """Build a well-formed char-level alignment for `text`, one char every N seconds."""
    chars = list(text)
    starts = [i * seconds_per_char for i in range(len(chars))]
    ends = [(i + 1) * seconds_per_char for i in range(len(chars))]
    return chars, starts, ends


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def dialogue_lines(path):
    return [l for l in read(path).splitlines() if l.startswith("Dialogue:")]


class TestWordsFromAlignment(unittest.TestCase):
    TEXT = "the bell rang twice"

    def test_splits_into_words_with_char_offsets(self):
        words = words_from_alignment(self.TEXT, *linear_alignment(self.TEXT))
        self.assertEqual([w["text"] for w in words], ["the", "bell", "rang", "twice"])
        for w in words:
            self.assertEqual(self.TEXT[w["cstart"]:w["cend"]], w["text"])

    def test_word_times_come_from_its_first_and_last_character(self):
        chars, starts, ends = linear_alignment(self.TEXT)
        words = words_from_alignment(self.TEXT, chars, starts, ends)
        for w in words:
            self.assertAlmostEqual(w["start"], starts[w["cstart"]], places=6)
            self.assertAlmostEqual(w["end"], ends[w["cend"] - 1], places=6)

    def test_timings_are_monotonic_and_non_overlapping(self):
        words = words_from_alignment(self.TEXT, *linear_alignment(self.TEXT))
        for a, b in zip(words, words[1:]):
            self.assertLessEqual(a["end"], b["start"])
            self.assertLess(a["start"], a["end"])

    def test_whitespace_runs_do_not_produce_empty_words(self):
        text = "  a   double   spaced   line  "
        words = words_from_alignment(text, *linear_alignment(text))
        self.assertEqual([w["text"] for w in words], ["a", "double", "spaced", "line"])

    def test_truncated_alignment_is_tolerated(self):
        """A short/ragged alignment must degrade, not raise — it comes from an ASR model."""
        chars, starts, ends = linear_alignment(self.TEXT)
        words = words_from_alignment(self.TEXT, chars[:8], starts[:8], ends[:8])
        self.assertEqual([w["text"] for w in words], ["the", "bell"])

    def test_empty_alignment_yields_no_words(self):
        self.assertEqual(words_from_alignment("", [], [], []), [])


class TestAssHelpers(unittest.TestCase):
    def test_timecode_format(self):
        self.assertEqual(_fmt(0), "0:00:00.00")
        self.assertEqual(_fmt(61.5), "0:01:01.50")
        self.assertEqual(_fmt(3661.25), "1:01:01.25")

    def test_negative_time_is_clamped(self):
        self.assertEqual(_fmt(-5), "0:00:00.00")

    def test_escape_neutralises_ass_control_characters(self):
        # Unescaped braces would be parsed as an ASS override block and swallow the text.
        self.assertNotIn("{", _esc("a {tag} b"))
        self.assertNotIn("}", _esc("a {tag} b"))
        self.assertNotIn("\n", _esc("two\nlines"))


class TestBuildAssRolling(unittest.TestCase):
    TEXT = "one two three four five six seven"

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._dir.name, "subs.ass")
        self.words = words_from_alignment(self.TEXT, *linear_alignment(self.TEXT, 0.1))
        build_ass_rolling({"words": self.words}, self.path)
        self.content = read(self.path)
        self.dialogues = dialogue_lines(self.path)

    def tearDown(self):
        self._dir.cleanup()

    def test_writes_a_wellformed_ass_file(self):
        for section in ("[Script Info]", "[V4+ Styles]", "[Events]"):
            self.assertIn(section, self.content)
        self.assertIn("PlayResX: 1080", self.content)
        self.assertIn("PlayResY: 1920", self.content)

    def test_one_dialogue_line_per_word_state(self):
        """Each word gets its own line so exactly one word can be highlighted at a time."""
        self.assertEqual(len(self.dialogues), len(self.words))

    def test_each_line_shows_at_most_the_window_of_words(self):
        for line in self.dialogues:
            text = line.split(",", 9)[-1]
            plain = re.sub(r"\{[^}]*\}", "", text)
            self.assertLessEqual(len(plain.split()), SUB_WINDOW)

    def test_exactly_one_word_is_highlighted_per_line(self):
        # The highlight is white (&H00FFFFFF) + an \fscx scale-up override.
        for line in self.dialogues:
            self.assertEqual(line.count(r"\fscx140"), 1, f"expected one popped word in: {line}")

    def test_lines_abut_without_gaps_or_overlap(self):
        times = []
        for line in self.dialogues:
            _, start, end = line.split(",", 3)[:3]
            times.append((start, end))
        for (_, end), (start, _) in zip(times, times[1:]):
            self.assertEqual(end, start, "a gap or overlap between states would flicker on screen")

    def test_first_line_starts_at_the_first_word(self):
        self.assertIn(_fmt(self.words[0]["start"]), self.dialogues[0])

    def test_single_word_script_does_not_crash(self):
        words = words_from_alignment("alone", *linear_alignment("alone"))
        out = os.path.join(self._dir.name, "one.ass")
        build_ass_rolling({"words": words}, out)
        self.assertEqual(len(dialogue_lines(out)), 1)

    def test_accent_colour_override_is_applied(self):
        out = os.path.join(self._dir.name, "accent.ass")
        build_ass_rolling({"words": self.words}, out, accent_hex="#FF0000")
        self.assertIn("&H000000FF", read(out))  # ASS is BGR


if __name__ == "__main__":
    unittest.main(verbosity=2)
