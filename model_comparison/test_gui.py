"""测试模型对比模块启动"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton

class TestComparisonGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("模型对比测试")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("模型对比模块测试"))
        layout.addWidget(QPushButton("测试按钮"))

if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    window = TestComparisonGUI()
    window.show()
    sys.exit(app.exec_())
