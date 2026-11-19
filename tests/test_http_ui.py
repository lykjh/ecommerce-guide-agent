"""让Python总回归同时覆盖真实chat.js事件处理器；需要Node.js。"""
from pathlib import Path
import subprocess
import unittest


class HttpUiTests(unittest.TestCase):
    def test_page_event_handlers(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            ["node", str(root / "tests/development/identity_session/identity_ui.test.mjs")], cwd=root,
            capture_output=True, text=True, encoding="utf-8", timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
