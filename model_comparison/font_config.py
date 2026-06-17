#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared Chinese font configuration helpers for model comparison.

This is migrated from the legacy main application utilities so the comparison
module can stay self-contained after the old folder is removed.
"""

from __future__ import annotations

import warnings
from typing import List, Optional


_PREFERRED_CHINESE_FONTS = [
    "Microsoft YaHei",
    "Microsoft JhengHei",
    "SimHei",
    "STSong",
    "STHeiti",
    "KaiTi",
    "SimSun",
    "NSimSun",
    "FangSong",
    "Arial Unicode MS",
    "DejaVu Sans",
    "Liberation Sans",
]


def _get_available_fonts() -> List[str]:
    import matplotlib.font_manager as fm

    return [font.name for font in fm.fontManager.ttflist]


def select_preferred_chinese_font() -> Optional[str]:
    """Return the first installed font from the preferred Chinese font list."""

    available_fonts = set(_get_available_fonts())
    for font_name in _PREFERRED_CHINESE_FONTS:
        if font_name in available_fonts:
            return font_name
    return None


def setup_matplotlib_chinese_font(module_name: str = "未知模块") -> Optional[str]:
    """Configure Matplotlib so Chinese text renders cleanly when possible."""

    try:
        import matplotlib.pyplot as plt

        selected_font = select_preferred_chinese_font()
        if selected_font:
            plt.rcParams["font.sans-serif"] = [selected_font] + list(_PREFERRED_CHINESE_FONTS)
            plt.rcParams["font.family"] = ["sans-serif"]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"{module_name}: 使用中文字体: {selected_font}")
        else:
            plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "sans-serif"]
            plt.rcParams["font.family"] = ["sans-serif"]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"{module_name}: 未找到合适的中文字体，使用默认字体")

        warnings.filterwarnings("ignore", category=UserWarning, message=".*Glyph.*missing from font.*")
        warnings.filterwarnings("ignore", category=UserWarning, message=".*findfont.*")
        return selected_font
    except Exception as exc:
        print(f"{module_name}: 配置字体失败: {exc}")
        return None


def get_available_chinese_fonts() -> List[str]:
    """Return the list of Chinese-capable fonts present on this machine."""

    try:
        available_fonts = set(_get_available_fonts())
        chinese_font_names = [
            "Microsoft YaHei",
            "Microsoft JhengHei",
            "SimHei",
            "STSong",
            "STHeiti",
            "KaiTi",
            "SimSun",
            "NSimSun",
            "FangSong",
            "Arial Unicode MS",
            "WenQuanYi Micro Hei",
            "Hiragino Sans GB",
        ]
        return [font for font in chinese_font_names if font in available_fonts]
    except Exception as exc:
        print(f"获取可用中文字体失败: {exc}")
        return []


def setup_qt_chinese_font(app, module_name: str = "未知模块") -> Optional[str]:
    """Apply a Chinese-capable font to a running Qt application if available."""

    try:
        from PyQt5.QtGui import QFont, QFontDatabase

        if app is None:
            return None

        available_families = set(QFontDatabase().families())
        selected_font = None
        for font_name in _PREFERRED_CHINESE_FONTS:
            if font_name in available_families:
                selected_font = font_name
                break

        if selected_font:
            app.setFont(QFont(selected_font))
            print(f"{module_name}: Qt 使用中文字体: {selected_font}")
        return selected_font
    except Exception as exc:
        print(f"{module_name}: 配置 Qt 字体失败: {exc}")
        return None


def setup_all_chinese_fonts(app=None, module_name: str = "未知模块") -> Optional[str]:
    """Configure both Matplotlib and Qt with a Chinese-capable font."""

    selected_font = setup_matplotlib_chinese_font(module_name)
    qt_font = setup_qt_chinese_font(app, module_name)
    return qt_font or selected_font


def setup_seaborn_chinese_font() -> bool:
    """Configure seaborn to use a Chinese-capable font when seaborn is available."""

    try:
        import seaborn as sns

        available_fonts = get_available_chinese_fonts()
        if available_fonts:
            sns.set_style("whitegrid")
            sns.set(font=available_fonts[0])
            return True
        return False
    except ImportError:
        return False
    except Exception as exc:
        print(f"配置 seaborn 字体失败: {exc}")
        return False
