from __future__ import annotations

import sys
import traceback
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget
except Exception:  # pragma: no cover - optional GUI dependency in lightweight environments
    QApplication = None
    QLabel = None
    QPushButton = None
    QTextEdit = None
    QVBoxLayout = None
    QWidget = None

try:
    from 代码.model_comparison.font_config import setup_all_chinese_fonts
except Exception:  # pragma: no cover - allow package-relative import when needed
    try:
        from .font_config import setup_all_chinese_fonts
    except Exception:  # pragma: no cover - keep launcher usable even if fonts helper is missing
        setup_all_chinese_fonts = None


if QWidget is not None:
    class BridgeWindow(QWidget):
        """Fallback window shown when the full comparison GUI cannot be loaded."""

        def __init__(self, error_text=""):
            super().__init__()
            self.setWindowTitle("模型对比模块")
            self.resize(900, 600)

            layout = QVBoxLayout(self)
            title = QLabel("模型对比模块已接入主系统")
            title.setStyleSheet("font-size: 20px; font-weight: bold;")
            layout.addWidget(title)

            hint = QLabel(
                "当前入口已经打通。若部分底层依赖尚未补齐，先保留占位窗口，后续可以继续把各个按钮逐步接到主系统里。"
            )
            hint.setWordWrap(True)
            layout.addWidget(hint)

            self.error_box = QTextEdit()
            self.error_box.setReadOnly(True)
            self.error_box.setPlaceholderText("如果完整界面加载失败，这里会显示原因。")
            self.error_box.setPlainText(error_text or "暂无错误信息。")
            layout.addWidget(self.error_box, 1)

            retry_btn = QPushButton("重新尝试加载完整界面")
            retry_btn.clicked.connect(self._show_retry_message)
            layout.addWidget(retry_btn)

        def _show_retry_message(self):
            self.error_box.append("\n已保留占位窗口。等后续依赖补齐后，可以直接把这个入口切换为完整界面。")
else:
    BridgeWindow = None


def main():
    if QApplication is None:
        print("PyQt5 is not available in this environment, so the model comparison GUI cannot be opened.")
        print("The main-system button still points here, and the launcher will show the real GUI when PyQt5 is installed.")
        return 0

    app = QApplication(sys.argv)
    if setup_all_chinese_fonts is not None:
        try:
            setup_all_chinese_fonts(app, "模型对比模块")
        except Exception:
            pass
    window = None
    try:
        from 代码.model_comparison.comparison_gui import ModelComparisonGUI

        window = ModelComparisonGUI()
    except Exception:
        if BridgeWindow is None:
            print(traceback.format_exc())
            return 1
        window = BridgeWindow(traceback.format_exc())

    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
