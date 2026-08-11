import unittest
import sys
from unittest.mock import MagicMock, patch

class StreamlitMock(MagicMock):
    def columns(self, num, *args, **kwargs):
        if isinstance(num, int):
            return [MagicMock() for _ in range(num)]
        if isinstance(num, (list, tuple)):
            return [MagicMock() for _ in num]
        return [MagicMock(), MagicMock()]

    def tabs(self, tabs, *args, **kwargs):
        return [MagicMock() for _ in tabs]

    def text_input(self, label, value="", *args, **kwargs):
        return value

    def selectbox(self, label, options, *args, **kwargs):
        if options:
            return options[0]
        return "mock_model"

mock_st = StreamlitMock()

modules_patcher = patch.dict(
    sys.modules,
    {
        "streamlit": mock_st,
        "pandas": MagicMock(),
        "plotly": MagicMock(),
        "plotly.express": MagicMock(),
        "plotly.graph_objects": MagicMock(),
    }
)
modules_patcher.start()
import dashboard
modules_patcher.stop()

class TestDashboard(unittest.TestCase):
    def test_validate_model_name_valid(self):
        valid_names = [
            "llama2",
            "llama-2-7b",
            "meta-llama/Llama-2-7b-chat-hf",
            "model_v1.0",
            "model:latest",
            "my.model.name",
            "12345"
        ]
        for name in valid_names:
            with self.subTest(name=name):
                self.assertEqual(dashboard.validate_model_name(name), name)

    def test_validate_model_name_invalid(self):
        invalid_names = [
            "",
            None,
            "model name with spaces",
            "model@name",
            "model!name",
            "model#name",
            "model$name",
            "model%name",
            "model^name",
            "model&name",
            "model*name",
            "model(name)",
            "model+",
            "model=",
            "model<",
            "model>",
            "model?",
            "model`",
            "model~",
            "model{",
            "model}",
            "model[",
            "model]"
        ]
        for name in invalid_names:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    dashboard.validate_model_name(name)

if __name__ == "__main__":
    unittest.main()
