import unittest
from advanced_benchmarks import is_safe_code, generate_filler_text

EXPECTED_DISTRACTORS = {
    "The software architecture patterns dictate that services must be decoupled.",
    "Quantum computing relies on superposition and entanglement to perform computations.",
    "A database transaction must satisfy the ACID properties to ensure reliability.",
    "Deep learning models require optimization algorithms like Adam or SGD to converge.",
    "The history of web browsers is characterized by intense competition and standardization.",
    "Distributed systems face challenges like network partitions, latency, and consensus protocols.",
    "Compiler design involves lexical analysis, parsing, semantic analysis, and code generation.",
    "Operating systems manage system resources, hardware devices, and process scheduling.",
    "Garbage collection algorithms reclaim memory occupied by objects that are no longer in use.",
    "Regular expressions are powerful tools for pattern matching and text manipulation."
}

class TestGenerateFillerText(unittest.TestCase):
    def test_output_generation_and_non_emptiness(self):
        paragraphs = generate_filler_text(target_tokens=100)
        self.assertIsInstance(paragraphs, list)
        self.assertGreater(len(paragraphs), 0)
        for paragraph in paragraphs:
            self.assertIsInstance(paragraph, str)
            self.assertGreater(len(paragraph.strip()), 0)

    def test_approximation_of_target_token_counts(self):
        token_targets = [10, 100, 1000]
        for target in token_targets:
            with self.subTest(target_tokens=target):
                paragraphs = generate_filler_text(target_tokens=target)
                expected_min_chars = target * 4.5
                total_chars = sum(len(p) + 1 for p in paragraphs)
                self.assertGreaterEqual(total_chars, expected_min_chars)

    def test_repetition_and_distractor_concatenation_logic(self):
        paragraphs = generate_filler_text(target_tokens=500)
        for paragraph in paragraphs:
            # Split paragraph into sentences by checking matching distractors
            sentences = []
            remaining = paragraph
            while remaining:
                matched = False
                for d in EXPECTED_DISTRACTORS:
                    if remaining.startswith(d):
                        sentences.append(d)
                        remaining = remaining[len(d):].lstrip()
                        matched = True
                        break
                if not matched:
                    self.fail(f"Could not parse valid distractor sentence from paragraph segment: {remaining[:50]}")
            self.assertEqual(len(sentences), 5, "Each paragraph should consist of exactly 5 distractor sentences")
            for sentence in sentences:
                self.assertIn(sentence, EXPECTED_DISTRACTORS)


class TestIsSafeCode(unittest.TestCase):
    def test_valid_code(self):
        code = "print('Hello World')\nx = 1 + 1"
        safe, msg = is_safe_code(code)
        self.assertTrue(safe)
        self.assertIsNone(msg)

    def test_syntax_error(self):
        code = "print('Hello World'"
        safe, msg = is_safe_code(code)
        self.assertFalse(safe)
        self.assertIn("Syntax error", msg)

    def test_forbidden_code_patterns(self):
        cases = [
            ("import subprocess", "import subprocess", "Forbidden"),
            ("import subprocess.Popen", "import sub-module", "Forbidden"),
            ("from subprocess import Popen", "from forbidden module", "Forbidden"),
            ("from os import system", "from forbidden name", "Forbidden"),
            ("import os\nos.system('ls')", "forbidden attribute usage", "Forbidden"),
            ("eval('1 + 1')", "forbidden identifier usage", "Forbidden"),
        ]
        for code, label, expected_keyword in cases:
            with self.subTest(case=label):
                safe, msg = is_safe_code(code)
                self.assertFalse(safe)
                self.assertIsNotNone(msg)
                self.assertIn(expected_keyword, msg)

if __name__ == "__main__":
    unittest.main()
