"""ThemeService: the application design system (dark default + light).

Every widget colour comes from a theme token; the canvas painter reads the
same tokens through ``ThemeService.tokens()`` so nodes, wires and chrome
always agree.
"""
from __future__ import annotations

THEMES: dict[str, dict] = {
    "dark": {
        "BG": "#17191d", "PANEL": "#1e2126", "SURFACE": "#16181c", "RAISED": "#262a31",
        "INPUT": "#24272d", "HOVER": "#2c3038", "BORDER": "#30343c", "BORDER_SOFT": "#272b32",
        "TEXT": "#e4e6ea", "TEXT_DIM": "#a0a6af", "TEXT_MUTED": "#6c727c",
        "ACCENT": "#4c8dff", "ACCENT_HOVER": "#6ba1ff", "ACCENT_TEXT": "#ffffff",
        "ACCENT_SOFT": "#1f3354",
        "SUCCESS": "#3fb950", "WARNING": "#e3a008", "ERROR": "#f85149",
        "CANVAS_BG": (22, 24, 28), "CANVAS_GRID": (34, 37, 43),
        "NODE_BODY": "#23262d", "NODE_BORDER": "#3a3f48", "NODE_TEXT": "#e4e6ea",
        "NODE_SUBTEXT": "#9aa0a9", "WIRE": "#5d6470",
    },
    "light": {
        "BG": "#eef0f3", "PANEL": "#ffffff", "SURFACE": "#ffffff", "RAISED": "#f6f7f9",
        "INPUT": "#f4f5f7", "HOVER": "#e9ecf0", "BORDER": "#d7dbe1", "BORDER_SOFT": "#e4e7eb",
        "TEXT": "#1d2127", "TEXT_DIM": "#59616c", "TEXT_MUTED": "#8b929c",
        "ACCENT": "#2f6fe4", "ACCENT_HOVER": "#4a83ea", "ACCENT_TEXT": "#ffffff",
        "ACCENT_SOFT": "#dce8fd",
        "SUCCESS": "#1f883d", "WARNING": "#b07a00", "ERROR": "#cf222e",
        "CANVAS_BG": (247, 248, 250), "CANVAS_GRID": (236, 238, 242),
        "NODE_BODY": "#ffffff", "NODE_BORDER": "#cfd4db", "NODE_TEXT": "#1d2127",
        "NODE_SUBTEXT": "#636b76", "WIRE": "#9aa1ab",
    },
}

# chart series colours, in order; web/src/theme.ts has the same list (tests/test_ui_polish.py)
SERIES = ("#4c8dff", "#f0883e", "#3fb950", "#d2a8ff", "#e3a008", "#56d4dd", "#ff7b72", "#9aa0a9")

DEFAULT_THEME = "dark"
MONO = "'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace"


def _build_qss(t: dict) -> str:
    return f"""
* {{ outline: none; }}
QWidget {{ background: {t['BG']}; color: {t['TEXT']}; font-size: 13px; }}
QLabel, QCheckBox, QRadioButton {{ background: transparent; }}
QMainWindow::separator {{ background: {t['BORDER_SOFT']}; width: 1px; height: 1px; }}
QMainWindow::separator:hover {{ background: {t['ACCENT']}; }}

/* ---------------------------------------------------------------- docks */
QDockWidget {{ titlebar-close-icon: none; color: {t['TEXT_DIM']}; font-size: 11px;
    font-weight: 600; }}
QDockWidget::title {{ background: {t['PANEL']}; padding: 7px 10px;
    border-bottom: 1px solid {t['BORDER_SOFT']}; text-align: left; }}
QDockWidget > QWidget {{ background: {t['PANEL']}; }}
QWidget#dockBody {{ background: {t['PANEL']}; }}

QWidget#toolbarSpacer {{ background: transparent; }}
QMainWindow > QTabBar::tab {{ min-width: 84px; padding: 6px 14px; }}

/* -------------------------------------------------------------- toolbar */
QToolBar {{ background: {t['PANEL']}; border: none; border-bottom: 1px solid {t['BORDER_SOFT']};
    padding: 4px 8px; spacing: 2px; }}
QToolBar::separator {{ background: {t['BORDER']}; width: 1px; margin: 6px 6px; }}
QToolButton {{ background: transparent; color: {t['TEXT']}; border: 1px solid transparent;
    border-radius: 6px; padding: 4px 8px; }}
QToolButton:hover {{ background: {t['HOVER']}; }}
QToolButton:pressed, QToolButton:checked {{ background: {t['ACCENT_SOFT']}; }}
QToolButton:disabled {{ color: {t['TEXT_MUTED']}; }}
QToolButton::menu-indicator {{ image: none; width: 0; }}
QToolButton#primaryAction {{ background: {t['ACCENT']}; color: {t['ACCENT_TEXT']};
    font-weight: 600; padding: 4px 12px; }}
QToolButton#primaryAction:hover {{ background: {t['ACCENT_HOVER']}; }}
QToolButton#primaryAction:disabled {{ background: {t['HOVER']}; color: {t['TEXT_MUTED']}; }}
QToolButton#stopAction:enabled {{ color: {t['ERROR']}; }}

/* --------------------------------------------------------------- inputs */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit#paramText {{
    background: {t['INPUT']}; color: {t['TEXT']}; border: 1px solid {t['BORDER']};
    border-radius: 5px; padding: 4px 7px; min-height: 18px;
    selection-background-color: {t['ACCENT']}; selection-color: {t['ACCENT_TEXT']}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {t['ACCENT']}; }}
QLineEdit:disabled, QSpinBox:disabled, QComboBox:disabled {{ color: {t['TEXT_MUTED']}; }}
QLineEdit#projectName {{ background: transparent; border: 1px solid transparent;
    font-weight: 600; min-width: 160px; }}
QLineEdit#projectName:hover {{ border-color: {t['BORDER']}; }}
QLineEdit#projectName:focus {{ background: {t['INPUT']}; border-color: {t['ACCENT']}; }}
QLineEdit#searchField {{ padding-left: 8px; }}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button,
QDoubleSpinBox::down-button {{ width: 0; border: none; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{ background: {t['RAISED']}; border: 1px solid {t['BORDER']};
    selection-background-color: {t['ACCENT']}; selection-color: {t['ACCENT_TEXT']};
    outline: none; padding: 2px; }}
QCheckBox {{ spacing: 7px; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border-radius: 3px;
    border: 1px solid {t['BORDER']}; background: {t['INPUT']}; }}
QCheckBox::indicator:checked {{ background: {t['ACCENT']}; border-color: {t['ACCENT']}; }}

QPushButton {{ background: {t['RAISED']}; color: {t['TEXT']}; border: 1px solid {t['BORDER']};
    border-radius: 6px; padding: 5px 12px; min-height: 18px; }}
QPushButton:hover {{ background: {t['HOVER']}; }}
QPushButton:pressed {{ background: {t['ACCENT_SOFT']}; }}
QPushButton:disabled {{ color: {t['TEXT_MUTED']}; background: {t['PANEL']}; }}
QPushButton#primaryButton {{ background: {t['ACCENT']}; border-color: {t['ACCENT']};
    color: {t['ACCENT_TEXT']}; font-weight: 600; }}
QPushButton#primaryButton:hover {{ background: {t['ACCENT_HOVER']}; }}
QPushButton#linkButton {{ background: transparent; border: none; color: {t['ACCENT']};
    padding: 2px 4px; }}

/* ---------------------------------------------------------- item views */
QTreeView, QTreeWidget, QListView, QListWidget, QTableView, QTableWidget {{
    background: {t['PANEL']}; alternate-background-color: {t['RAISED']};
    border: none; selection-background-color: {t['ACCENT_SOFT']};
    selection-color: {t['TEXT']}; show-decoration-selected: 1; }}
QTreeView::item, QListView::item {{ padding: 3px 4px; border: none; }}
QTreeView::item:hover, QListView::item:hover {{ background: {t['HOVER']}; }}
QTreeView::item:selected, QListView::item:selected {{ background: {t['ACCENT_SOFT']};
    color: {t['TEXT']}; }}
QHeaderView::section {{ background: {t['PANEL']}; color: {t['TEXT_DIM']}; border: none;
    border-bottom: 1px solid {t['BORDER_SOFT']}; padding: 5px 8px; font-size: 11px;
    font-weight: 600; }}
QTableView {{ gridline-color: {t['BORDER_SOFT']}; }}

/* ----------------------------------------------------------------- tabs */
QTabWidget::pane {{ border: none; background: {t['PANEL']}; }}
QTabBar {{ background: {t['PANEL']}; }}
QTabBar::tab {{ background: transparent; color: {t['TEXT_DIM']}; padding: 7px 12px;
    border: none; border-bottom: 2px solid transparent; font-size: 12px; }}
QTabBar::tab:hover {{ color: {t['TEXT']}; }}
QTabBar::tab:selected {{ color: {t['TEXT']}; border-bottom: 2px solid {t['ACCENT']}; }}

/* ------------------------------------------------------- text surfaces */
QPlainTextEdit, QTextEdit, QTextBrowser {{ background: {t['SURFACE']}; color: {t['TEXT']};
    border: none; selection-background-color: {t['ACCENT']};
    selection-color: {t['ACCENT_TEXT']}; }}
QPlainTextEdit#codeView, QPlainTextEdit#logView {{ font-family: {MONO}; font-size: 12px; }}

/* ---------------------------------------------------------------- menus */
QMenuBar {{ background: {t['PANEL']}; border-bottom: 1px solid {t['BORDER_SOFT']}; }}
QMenuBar::item {{ padding: 4px 9px; background: transparent; }}
QMenuBar::item:selected {{ background: {t['HOVER']}; border-radius: 4px; }}
QMenu {{ background: {t['RAISED']}; border: 1px solid {t['BORDER']}; padding: 4px;
    border-radius: 8px; }}
QMenu::item {{ padding: 6px 26px 6px 12px; border-radius: 5px; }}
QMenu::item:selected {{ background: {t['ACCENT']}; color: {t['ACCENT_TEXT']}; }}
QMenu::item:disabled {{ color: {t['TEXT_MUTED']}; }}
QMenu::separator {{ height: 1px; background: {t['BORDER']}; margin: 4px 8px; }}
QMenu::icon {{ padding-left: 6px; }}

/* ------------------------------------------------------------ chrome */
QStatusBar {{ background: {t['PANEL']}; color: {t['TEXT_DIM']};
    border-top: 1px solid {t['BORDER_SOFT']}; font-size: 12px; }}
QStatusBar::item {{ border: none; }}
QLabel#statusChip {{ color: {t['TEXT_DIM']}; padding: 0 8px; }}
QLabel#statusChip[state="error"] {{ color: {t['ERROR']}; }}
QLabel#statusChip[state="warning"] {{ color: {t['WARNING']}; }}
QLabel#statusChip[state="ok"] {{ color: {t['SUCCESS']}; }}
QLabel#sectionTitle {{ color: {t['TEXT_DIM']}; font-size: 11px; font-weight: 600;
    padding: 6px 0 2px 0; }}
QLabel#blockTitle {{ font-size: 15px; font-weight: 600; }}
QLabel#blockMeta {{ color: {t['TEXT_DIM']}; font-size: 12px; }}
QLabel#errorText {{ color: {t['ERROR']}; font-size: 12px; }}
QLabel#blockDesc {{ color: {t['TEXT_DIM']}; }}
QLabel#blockWhy {{ color: {t['TEXT']}; border-left: 2px solid {t['ACCENT']}; padding: 2px 0 2px 8px; }}
QLabel#paramName {{ color: {t['TEXT_DIM']}; }}
QLabel#emptyState {{ color: {t['TEXT_MUTED']}; padding: 24px; }}
QLabel#badge {{ background: {t['RAISED']}; color: {t['TEXT_DIM']};
    border: 1px solid {t['BORDER']}; border-radius: 9px; padding: 1px 8px; font-size: 11px; }}
QLabel#metricValue {{ font-size: 20px; font-weight: 600; }}
QLabel#metricName {{ color: {t['TEXT_DIM']}; font-size: 11px; }}
QFrame#metricTile {{ background: {t['RAISED']}; border: 1px solid {t['BORDER_SOFT']};
    border-radius: 8px; }}
QFrame#issueBanner {{ border-radius: 6px; padding: 4px; }}
QFrame#hline {{ background: {t['BORDER_SOFT']}; max-height: 1px; border: none; }}

QSplitter::handle {{ background: {t['BORDER_SOFT']}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {t['BORDER']}; border-radius: 4px;
    min-height: 24px; margin: 2px; }}
QScrollBar::handle:vertical:hover {{ background: {t['TEXT_MUTED']}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {t['BORDER']}; border-radius: 4px;
    min-width: 24px; margin: 2px; }}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{
    width: 0; height: 0; background: none; }}

QProgressBar {{ background: {t['INPUT']}; border: none; border-radius: 2px;
    max-height: 4px; min-height: 4px; }}
QProgressBar::chunk {{ background: {t['ACCENT']}; border-radius: 2px; }}

QToolTip {{ background: {t['RAISED']}; color: {t['TEXT']}; border: 1px solid {t['BORDER']};
    border-radius: 6px; padding: 6px 8px; }}

QDialog {{ background: {t['PANEL']}; }}
QDialog#cmdPalette {{ background: {t['RAISED']}; border: 1px solid {t['BORDER']};
    border-radius: 10px; }}
QLineEdit#cmdInput {{ background: transparent; border: none;
    border-bottom: 1px solid {t['BORDER']}; border-radius: 0; font-size: 15px;
    padding: 12px 14px; }}
QListWidget#cmdList {{ background: transparent; padding: 4px; }}
QListWidget#cmdList::item {{ padding: 7px 10px; border-radius: 5px; }}
QListWidget#cmdList::item:selected {{ background: {t['ACCENT']}; color: {t['ACCENT_TEXT']}; }}
QLabel#toast {{ background: {t['RAISED']}; color: {t['TEXT']}; border: 1px solid {t['BORDER']};
    border-radius: 8px; padding: 8px 14px; }}
"""


class ThemeService:
    """Owns the active theme; ``apply`` can be called again to switch live."""

    def __init__(self) -> None:
        self._active = DEFAULT_THEME

    def active(self) -> str:
        return self._active

    def names(self) -> list[str]:
        return list(THEMES)

    def tokens(self) -> dict:
        return THEMES[self._active]

    def canvas_colors(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        t = THEMES[self._active]
        return t["CANVAS_BG"], t["CANVAS_GRID"]

    def apply(self, app, name: str = DEFAULT_THEME) -> None:
        from PySide6 import QtGui

        self._active = name if name in THEMES else DEFAULT_THEME
        t = THEMES[self._active]
        if not app.property("aime_fusion"):   # once: a new style deletes the old one,
            app.setStyle("Fusion")            # which widgets still alive may use
            app.setProperty("aime_fusion", True)
        palette = QtGui.QPalette()
        for role, key in ((QtGui.QPalette.ColorRole.Window, "BG"),
                          (QtGui.QPalette.ColorRole.WindowText, "TEXT"),
                          (QtGui.QPalette.ColorRole.Base, "PANEL"),
                          (QtGui.QPalette.ColorRole.AlternateBase, "RAISED"),
                          (QtGui.QPalette.ColorRole.Text, "TEXT"),
                          (QtGui.QPalette.ColorRole.Button, "RAISED"),
                          (QtGui.QPalette.ColorRole.ButtonText, "TEXT"),
                          (QtGui.QPalette.ColorRole.Highlight, "ACCENT"),
                          (QtGui.QPalette.ColorRole.HighlightedText, "ACCENT_TEXT"),
                          (QtGui.QPalette.ColorRole.ToolTipBase, "RAISED"),
                          (QtGui.QPalette.ColorRole.ToolTipText, "TEXT"),
                          (QtGui.QPalette.ColorRole.PlaceholderText, "TEXT_MUTED")):
            palette.setColor(role, QtGui.QColor(t[key]))
        app.setPalette(palette)
        app.setStyleSheet(_build_qss(t))
        from ai_made_easy.ui import icons

        icons.set_color(t["TEXT_DIM"], t["ACCENT_TEXT"])


def apply_dark_theme(app) -> None:  # back-compat for scripts
    ThemeService().apply(app, DEFAULT_THEME)
