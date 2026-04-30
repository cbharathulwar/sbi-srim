import sys
import os
import json
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSpinBox, QSlider, QStackedWidget,
    QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox, QFormLayout,
    QMessageBox, QFrame
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

plt.style.use('dark_background')

# ===============================
# Config & Paths
# ===============================
RESULTS_CSV = "results/gvp_egnn_v2/eval_results.csv"
RAW_DATA_CSV = "data/mcpe3d/mcpe_3d_eval.csv"
LEADERBOARD_FILE = "results/gvp_egnn_v2/leaderboard.json"

# Filter: only show tracks above this energy so the game is a fair/showcase fight.
MIN_ENERGY_KEV = 5.0
MAX_ENERGY_KEV = 100.0

# Energy slider uses a log scale 0-1000 mapping to MIN_ENERGY_KEV..MAX_ENERGY_KEV
ENERGY_SLIDER_RES = 1000

# ===============================
# Theme
# ===============================
MODERN_THEME = """
QWidget {
    background-color: #0f1419;
    color: #e0e0e0;
    font-family: -apple-system, 'SF Pro Display', 'Segoe UI', sans-serif;
    font-size: 14px;
}
QGroupBox {
    border: 2px solid #1f4e5a;
    border-radius: 10px;
    margin-top: 18px;
    font-weight: bold;
    font-size: 15px;
    color: #00d4e0;
    padding-top: 12px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 8px;
    background-color: #0f1419;
}
QPushButton {
    background-color: #1f2937;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 12px;
    font-weight: bold;
    font-size: 14px;
    color: #e0e0e0;
}
QPushButton:hover {
    background-color: #2d3748;
    border: 1px solid #00d4e0;
    color: #ffffff;
}
QPushButton:pressed {
    background-color: #00d4e0;
    color: #000000;
}
QPushButton:disabled {
    background-color: #1a1f24;
    color: #4a5568;
    border: 1px solid #2d3748;
}
QLineEdit, QSpinBox {
    background-color: #1f2937;
    border: 1px solid #374151;
    padding: 8px;
    border-radius: 6px;
    color: white;
    font-size: 15px;
}
QSlider::groove:horizontal {
    border: 1px solid #2d3748;
    height: 10px;
    background: #1a1f24;
    border-radius: 5px;
}
QSlider::handle:horizontal {
    background: #00d4e0;
    width: 22px;
    margin: -7px 0;
    border-radius: 11px;
    border: 2px solid #0f1419;
}
QSlider::handle:horizontal:hover {
    background: #4be6f0;
    width: 24px;
    margin: -8px 0;
    border-radius: 12px;
}
QSlider::sub-page:horizontal {
    background: #00d4e0;
    border-radius: 5px;
}
QTableWidget {
    background-color: #1f2937;
    border: none;
    gridline-color: #2d3748;
    font-size: 15px;
}
QHeaderView::section {
    background-color: #0f1419;
    padding: 8px;
    border: 1px solid #2d3748;
    font-weight: bold;
    color: #00d4e0;
}
QLabel { color: #e0e0e0; }
"""


def slider_to_energy(s):
    """Map slider value [0, ENERGY_SLIDER_RES] to keV [MIN, MAX] on log scale."""
    frac = s / ENERGY_SLIDER_RES
    log_min = np.log(MIN_ENERGY_KEV)
    log_max = np.log(MAX_ENERGY_KEV)
    return float(np.exp(log_min + frac * (log_max - log_min)))


def energy_to_slider(e):
    """Inverse of slider_to_energy."""
    e = max(MIN_ENERGY_KEV, min(MAX_ENERGY_KEV, e))
    log_min = np.log(MIN_ENERGY_KEV)
    log_max = np.log(MAX_ENERGY_KEV)
    return int(ENERGY_SLIDER_RES * (np.log(e) - log_min) / (log_max - log_min))


def vacancy_energy_hint(n_vac):
    """Rough energy hint based on vacancy count (calibrated from training data)."""
    if n_vac < 25:
        return "Low energy (likely 5-15 keV)"
    elif n_vac < 60:
        return "Medium energy (likely 10-40 keV)"
    elif n_vac < 120:
        return "High energy (likely 30-70 keV)"
    else:
        return "Very high energy (likely 50-100 keV)"


# ===============================
# Main Application
# ===============================
class VectorGame(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Human vs. AI: 3D Track Challenge")
        self.resize(1400, 950)
        self.setStyleSheet(MODERN_THEME)

        self.load_data()
        self.load_leaderboard()

        # Game State Variables
        self.round_tracks = []
        self.current_track_idx = 0
        self.player_name = ""
        self.total_score = 0
        self.ai_total_score = 0
        self.locked_in = False
        self.first_track_shown = False

        # Build UI Screens
        self.stacked = QStackedWidget(self)
        self.build_start_screen()
        self.build_game_screen()
        self.build_leaderboard_screen()

        self.stacked.addWidget(self.start_screen)
        self.stacked.addWidget(self.game_screen)
        self.stacked.addWidget(self.lb_screen)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.stacked)

        self.stacked.setCurrentWidget(self.start_screen)

    def load_data(self):
        print("[INFO] Loading game data...")
        self.df_results = pd.read_csv(RESULTS_CSV)
        df_raw = pd.read_csv(RAW_DATA_CSV)

        df_raw.sort_values('ion_number', inplace=True)
        df_raw.reset_index(drop=True, inplace=True)

        pts_list = []
        for ion_num, group in df_raw.groupby('ion_number', sort=True):
            pts = group[['x', 'y', 'z']].values
            if len(pts) < 3:
                continue
            pts = pts - np.mean(pts, axis=0)
            pts_list.append(pts)

        if len(pts_list) != len(self.df_results):
            print(f"[WARN] Track count mismatch: {len(pts_list)} raw vs "
                  f"{len(self.df_results)} eval — truncating to min.")
            n = min(len(pts_list), len(self.df_results))
            pts_list = pts_list[:n]
            self.df_results = self.df_results.iloc[:n].reset_index(drop=True)

        self.df_results['raw_points'] = pts_list

        before = len(self.df_results)
        mask = ((self.df_results['true_energy'] >= MIN_ENERGY_KEV) &
                (self.df_results['true_energy'] <= MAX_ENERGY_KEV))
        self.df_results = self.df_results[mask].reset_index(drop=True)
        print(f"[INFO] Loaded {before} tracks, kept {len(self.df_results)} in "
              f"[{MIN_ENERGY_KEV}, {MAX_ENERGY_KEV}] keV")

    def load_leaderboard(self):
        if os.path.exists(LEADERBOARD_FILE):
            with open(LEADERBOARD_FILE, 'r') as f:
                self.leaderboard = json.load(f)
        else:
            self.leaderboard = []

    def save_leaderboard(self):
        self.leaderboard.sort(key=lambda x: x['score'], reverse=True)
        self.leaderboard = self.leaderboard[:10]
        os.makedirs(os.path.dirname(LEADERBOARD_FILE), exist_ok=True)
        with open(LEADERBOARD_FILE, 'w') as f:
            json.dump(self.leaderboard, f, indent=2)

    # =====================================================
    # SCREEN 1: START MENU
    # =====================================================
    def build_start_screen(self):
        self.start_screen = QWidget()
        layout = QVBoxLayout(self.start_screen)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)

        # Big title
        title = QLabel("HUMAN vs. AI")
        title.setStyleSheet("font-size: 64px; font-weight: 900; color: #00d4e0; "
                            "letter-spacing: 4px; margin-top: 40px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("3D Nuclear Recoil Direction Challenge")
        subtitle.setStyleSheet("font-size: 22px; color: #e0e0e0; "
                               "letter-spacing: 1px; margin-bottom: 10px;")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)

        # Description box
        desc = QLabel(
            "<center>"
            "Look at a 3D point cloud of crystal damage.<br>"
            "Guess the direction of the incoming particle and its energy.<br><br>"
            "<span style='color:#00d4e0; font-weight:bold;'>Can you beat the neural network?</span>"
            "</center>"
        )
        desc.setStyleSheet("font-size: 16px; color: #b0b0b0; padding: 18px; "
                           "background-color: #1a1f24; border-radius: 10px; "
                           "border: 1px solid #2d3748; max-width: 600px;")
        desc.setFixedWidth(600)
        layout.addWidget(desc, alignment=Qt.AlignCenter)

        # Form
        form_widget = QFrame()
        form_widget.setStyleSheet("QFrame { background-color: #1a1f24; "
                                  "border-radius: 10px; padding: 20px; }")
        form_widget.setFixedWidth(420)
        form = QFormLayout(form_widget)
        form.setSpacing(12)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter your name...")
        self.name_input.setFixedHeight(36)

        self.rounds_input = QSpinBox()
        self.rounds_input.setRange(1, 20)
        self.rounds_input.setValue(5)
        self.rounds_input.setFixedHeight(36)

        name_lbl = QLabel("Player Name:")
        name_lbl.setStyleSheet("font-size: 15px; font-weight: bold; color: #00d4e0;")
        rounds_lbl = QLabel("Tracks per Game:")
        rounds_lbl.setStyleSheet("font-size: 15px; font-weight: bold; color: #00d4e0;")

        form.addRow(name_lbl, self.name_input)
        form.addRow(rounds_lbl, self.rounds_input)

        layout.addWidget(form_widget, alignment=Qt.AlignCenter)

        start_btn = QPushButton("▶  START CHALLENGE")
        start_btn.setFixedSize(280, 56)
        start_btn.setStyleSheet(
            "QPushButton { background-color: #00d4e0; color: #000000; "
            "font-size: 18px; font-weight: bold; border-radius: 28px; "
            "letter-spacing: 2px; }"
            "QPushButton:hover { background-color: #4be6f0; }"
        )
        start_btn.clicked.connect(self.start_game)
        layout.addWidget(start_btn, alignment=Qt.AlignCenter)

        # Footnote
        footnote = QLabel(
            "Powered by GVP-EGNN  |  Ang. resolution: ~7°  |  Energy MAE: 5 keV"
        )
        footnote.setStyleSheet("font-size: 12px; color: #4a5568; margin-top: 20px;")
        footnote.setAlignment(Qt.AlignCenter)
        layout.addWidget(footnote)

    # =====================================================
    # SCREEN 2: GAMEPLAY
    # =====================================================
    def build_game_screen(self):
        self.game_screen = QWidget()
        outer = QHBoxLayout(self.game_screen)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(12)

        # ── Left Panel (Controls) ──
        left_panel = QWidget()
        left_panel.setFixedWidth(420)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)

        # Big HUD scoreboard
        hud_frame = QFrame()
        hud_frame.setStyleSheet("QFrame { background-color: #1a1f24; "
                                "border-radius: 12px; border: 2px solid #1f4e5a; "
                                "padding: 14px; }")
        hud_layout = QVBoxLayout(hud_frame)
        hud_layout.setSpacing(6)

        self.track_label = QLabel("Track 1 / 5")
        self.track_label.setStyleSheet("font-size: 22px; font-weight: bold; "
                                       "color: #00d4e0; letter-spacing: 1px;")
        self.track_label.setAlignment(Qt.AlignCenter)
        hud_layout.addWidget(self.track_label)

        score_row = QHBoxLayout()
        self.human_score_label = QLabel("YOU\n0")
        self.human_score_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #00ff88; "
            "background-color: #0f1419; padding: 10px; border-radius: 8px;"
        )
        self.human_score_label.setAlignment(Qt.AlignCenter)
        self.ai_score_label = QLabel("AI\n0")
        self.ai_score_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #ff5555; "
            "background-color: #0f1419; padding: 10px; border-radius: 8px;"
        )
        self.ai_score_label.setAlignment(Qt.AlignCenter)
        score_row.addWidget(self.human_score_label, stretch=1)
        score_row.addWidget(self.ai_score_label, stretch=1)
        hud_layout.addLayout(score_row)

        left_layout.addWidget(hud_frame)

        # Big primary hint
        self.hint_banner = QLabel(
            "💡  Click any blue dot to aim your arrow"
        )
        self.hint_banner.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #ffd700; "
            "background-color: #2d2410; padding: 12px; border-radius: 8px; "
            "border: 1px solid #5a4a10;"
        )
        self.hint_banner.setAlignment(Qt.AlignCenter)
        self.hint_banner.setWordWrap(True)
        left_layout.addWidget(self.hint_banner)

        # Direction controls
        dir_group = QGroupBox("1. AIM YOUR DIRECTION")
        dir_layout = QVBoxLayout()
        dir_layout.setSpacing(8)

        # Show numeric values
        elev_row = QHBoxLayout()
        elev_lbl = QLabel("Up/Down")
        elev_lbl.setStyleSheet("font-size: 13px; color: #a0a0a0;")
        self.elev_value_label = QLabel("0°")
        self.elev_value_label.setStyleSheet("font-size: 13px; color: #00d4e0; "
                                            "font-weight: bold;")
        self.elev_value_label.setAlignment(Qt.AlignRight)
        elev_row.addWidget(elev_lbl)
        elev_row.addWidget(self.elev_value_label)
        dir_layout.addLayout(elev_row)

        self.elev_slider = QSlider(Qt.Horizontal)
        self.elev_slider.setRange(-90, 90)
        self.elev_slider.valueChanged.connect(self.update_plot_live)
        self.elev_slider.valueChanged.connect(
            lambda v: self.elev_value_label.setText(f"{v}°"))
        dir_layout.addWidget(self.elev_slider)

        azim_row = QHBoxLayout()
        azim_lbl = QLabel("Left/Right")
        azim_lbl.setStyleSheet("font-size: 13px; color: #a0a0a0;")
        self.azim_value_label = QLabel("0°")
        self.azim_value_label.setStyleSheet("font-size: 13px; color: #00d4e0; "
                                            "font-weight: bold;")
        self.azim_value_label.setAlignment(Qt.AlignRight)
        azim_row.addWidget(azim_lbl)
        azim_row.addWidget(self.azim_value_label)
        dir_layout.addLayout(azim_row)

        self.azim_slider = QSlider(Qt.Horizontal)
        self.azim_slider.setRange(-180, 180)
        self.azim_slider.valueChanged.connect(self.update_plot_live)
        self.azim_slider.valueChanged.connect(
            lambda v: self.azim_value_label.setText(f"{v}°"))
        dir_layout.addWidget(self.azim_slider)

        dir_group.setLayout(dir_layout)
        left_layout.addWidget(dir_group)

        # Energy controls
        eng_group = QGroupBox("2. GUESS THE ENERGY")
        eng_layout = QVBoxLayout()
        eng_layout.setSpacing(8)

        self.energy_hint_label = QLabel("More vacancies = higher energy")
        self.energy_hint_label.setStyleSheet(
            "font-size: 12px; color: #a0a0a0; font-style: italic;")
        self.energy_hint_label.setWordWrap(True)
        eng_layout.addWidget(self.energy_hint_label)

        eng_row = QHBoxLayout()
        eng_label_text = QLabel("Energy:")
        eng_label_text.setStyleSheet("font-size: 13px; color: #a0a0a0;")
        self.energy_value_label = QLabel("10.0 keV")
        self.energy_value_label.setStyleSheet(
            "font-size: 18px; color: #00d4e0; font-weight: bold;")
        self.energy_value_label.setAlignment(Qt.AlignRight)
        eng_row.addWidget(eng_label_text)
        eng_row.addWidget(self.energy_value_label)
        eng_layout.addLayout(eng_row)

        self.energy_slider = QSlider(Qt.Horizontal)
        self.energy_slider.setRange(0, ENERGY_SLIDER_RES)
        self.energy_slider.setValue(energy_to_slider(10.0))
        self.energy_slider.valueChanged.connect(self.on_energy_slider)
        eng_layout.addWidget(self.energy_slider)

        # Energy scale ticks
        scale_row = QHBoxLayout()
        for tick in [5, 10, 30, 100]:
            t = QLabel(f"{tick}")
            t.setStyleSheet("font-size: 11px; color: #4a5568;")
            t.setAlignment(Qt.AlignCenter)
            scale_row.addWidget(t)
        eng_layout.addLayout(scale_row)

        eng_group.setLayout(eng_layout)
        left_layout.addWidget(eng_group)

        # Action buttons
        btn_row = QHBoxLayout()
        self.reset_btn = QPushButton("↺  Reset")
        self.reset_btn.setFixedHeight(48)
        self.reset_btn.setStyleSheet(
            "QPushButton { background-color: #2d3748; font-size: 14px; }"
            "QPushButton:hover { background-color: #4a5568; }"
        )
        self.reset_btn.clicked.connect(self.reset_aim)
        btn_row.addWidget(self.reset_btn, stretch=1)

        self.action_btn = QPushButton("LOCK IN GUESS")
        self.action_btn.setFixedHeight(48)
        self.action_btn.setStyleSheet(
            "QPushButton { background-color: #ef4444; color: white; "
            "font-size: 16px; font-weight: bold; }"
            "QPushButton:hover { background-color: #f87171; }"
        )
        self.action_btn.clicked.connect(self.handle_action)
        btn_row.addWidget(self.action_btn, stretch=2)

        left_layout.addLayout(btn_row)

        # Feedback panel
        self.feedback_label = QLabel("Analyze the track and lock in your guess.")
        self.feedback_label.setWordWrap(True)
        self.feedback_label.setStyleSheet(
            "font-size: 13px; padding: 14px; background-color: #1a1f24; "
            "border-radius: 8px; border: 1px solid #2d3748; color: #b0b0b0;"
        )
        self.feedback_label.setMinimumHeight(140)
        left_layout.addWidget(self.feedback_label)

        # Help button at bottom
        help_row = QHBoxLayout()
        self.info_btn = QPushButton("?  Help")
        self.info_btn.setFixedHeight(34)
        self.info_btn.setStyleSheet(
            "QPushButton { background-color: #1a1f24; color: #00d4e0; "
            "font-size: 12px; }"
        )
        self.info_btn.clicked.connect(self.show_instructions)
        help_row.addWidget(self.info_btn)
        left_layout.addLayout(help_row)

        left_layout.addStretch()
        outer.addWidget(left_panel)

        # ── Right Panel (3D Plot) ──
        plot_container = QFrame()
        plot_container.setStyleSheet(
            "QFrame { background-color: #0f1419; border-radius: 12px; "
            "border: 2px solid #1f4e5a; }"
        )
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(8, 8, 8, 8)

        self.figure = Figure(facecolor='#0f1419')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('pick_event', self.on_pick)
        plot_layout.addWidget(self.canvas)

        # Plot caption
        self.plot_caption = QLabel(
            "🖱  Drag plot to rotate camera   |   "
            "👆  Click a vacancy dot to aim arrow at it"
        )
        self.plot_caption.setStyleSheet(
            "font-size: 13px; color: #a0a0a0; padding: 6px;"
        )
        self.plot_caption.setAlignment(Qt.AlignCenter)
        plot_layout.addWidget(self.plot_caption)

        outer.addWidget(plot_container, stretch=1)

    # =====================================================
    # SCREEN 3: LEADERBOARD
    # =====================================================
    def build_leaderboard_screen(self):
        self.lb_screen = QWidget()
        layout = QVBoxLayout(self.lb_screen)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)

        title = QLabel("GAME OVER")
        title.setStyleSheet("font-size: 56px; font-weight: 900; color: #00d4e0; "
                            "letter-spacing: 6px; margin-top: 30px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self.final_score_label = QLabel("")
        self.final_score_label.setStyleSheet(
            "font-size: 24px; padding: 16px; background-color: #1a1f24; "
            "border-radius: 10px; border: 2px solid #1f4e5a;")
        self.final_score_label.setAlignment(Qt.AlignCenter)
        self.final_score_label.setFixedWidth(500)
        layout.addWidget(self.final_score_label, alignment=Qt.AlignCenter)

        lb_title = QLabel("🏆  TOP 10 LEADERBOARD")
        lb_title.setStyleSheet("font-size: 22px; font-weight: bold; "
                               "color: #ffd700; margin-top: 10px;")
        lb_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(lb_title)

        self.lb_table = QTableWidget()
        self.lb_table.setColumnCount(3)
        self.lb_table.setHorizontalHeaderLabels(["Rank", "Player", "Score"])
        self.lb_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.lb_table.setFixedWidth(560)
        self.lb_table.setFixedHeight(360)
        self.lb_table.verticalHeader().setVisible(False)
        layout.addWidget(self.lb_table, alignment=Qt.AlignCenter)

        replay_btn = QPushButton("▶  PLAY AGAIN")
        replay_btn.setFixedSize(280, 52)
        replay_btn.setStyleSheet(
            "QPushButton { background-color: #00d4e0; color: #000000; "
            "font-size: 16px; font-weight: bold; border-radius: 26px; "
            "letter-spacing: 2px; }"
            "QPushButton:hover { background-color: #4be6f0; }"
        )
        replay_btn.clicked.connect(lambda: self.stacked.setCurrentWidget(self.start_screen))
        layout.addWidget(replay_btn, alignment=Qt.AlignCenter)

    # =====================================================
    # Game Logic
    # =====================================================
    def show_instructions(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("How to Play")
        msg.setStyleSheet(
            "QLabel { color: white; min-width: 480px; font-size: 14px; }"
            "QMessageBox { background-color: #1a1f24; }"
            "QPushButton { background-color: #00d4e0; color: black; "
            "border-radius: 6px; padding: 8px 20px; font-weight: bold; }"
        )
        msg.setText(
            "<h2 style='color:#00d4e0;'>How to Play</h2>"
            "<p><b>1. Aim your arrow:</b><br>"
            "👆 <span style='color:#ffd700;'>Click any blue vacancy dot</span> on the 3D plot to snap your yellow arrow to it. "
            "Use the sliders for fine adjustments.</p>"
            "<p><b>2. Guess the energy:</b><br>"
            "Look at the number of vacancies — more vacancies generally means higher energy. "
            "Drag the energy slider to your guess.</p>"
            "<p><b>3. Lock it in:</b><br>"
            "Click the red <b>LOCK IN GUESS</b> button to see how you did vs the AI.</p>"
            "<h3 style='color:#ffd700;'>🏆 Scoring</h3>"
            "<ul>"
            "<li><b>Direction (1000 pts):</b> -11 points per degree of error</li>"
            "<li><b>Energy (1000 pts):</b> -20 points per keV of error</li>"
            "<li><b>Beat the AI bonus:</b> +250 if you beat AI's angle, +250 for energy</li>"
            "</ul>"
        )
        msg.exec_()

    def start_game(self):
        self.player_name = self.name_input.text().strip() or "Guest"
        num_rounds = self.rounds_input.value()

        self.round_tracks = self.df_results.sample(n=num_rounds).to_dict('records')
        self.current_track_idx = 0
        self.total_score = 0
        self.ai_total_score = 0
        self.first_track_shown = False

        self.stacked.setCurrentWidget(self.game_screen)
        self.load_track()

        # Auto-show instructions on first ever run
        if not self.first_track_shown:
            self.first_track_shown = True
            QTimer.singleShot(300, self.show_instructions)

    def reset_aim(self):
        if self.locked_in:
            return
        self.elev_slider.blockSignals(True)
        self.azim_slider.blockSignals(True)
        self.elev_slider.setValue(0)
        self.azim_slider.setValue(0)
        self.elev_slider.blockSignals(False)
        self.azim_slider.blockSignals(False)
        self.elev_value_label.setText("0°")
        self.azim_value_label.setText("0°")
        self.update_plot_live()

    def load_track(self):
        self.locked_in = False
        self.elev_slider.setEnabled(True)
        self.azim_slider.setEnabled(True)
        self.energy_slider.setEnabled(True)
        self.reset_btn.setEnabled(True)

        self.elev_slider.blockSignals(True)
        self.azim_slider.blockSignals(True)
        self.energy_slider.blockSignals(True)
        self.elev_slider.setValue(0)
        self.azim_slider.setValue(0)
        self.energy_slider.setValue(energy_to_slider(10.0))
        self.elev_slider.blockSignals(False)
        self.azim_slider.blockSignals(False)
        self.energy_slider.blockSignals(False)

        self.elev_value_label.setText("0°")
        self.azim_value_label.setText("0°")
        self.energy_value_label.setText("10.0 keV")

        # Update vacancy hint
        row = self.round_tracks[self.current_track_idx]
        n_vac = len(row['raw_points'])
        hint = vacancy_energy_hint(n_vac)
        self.energy_hint_label.setText(f"💡 {n_vac} vacancies → {hint}")

        self.action_btn.setText("LOCK IN GUESS")
        self.action_btn.setStyleSheet(
            "QPushButton { background-color: #ef4444; color: white; "
            "font-size: 16px; font-weight: bold; }"
            "QPushButton:hover { background-color: #f87171; }"
        )
        self.feedback_label.setText("Analyze the track and lock in your guess.")
        self.update_hud()
        self.init_plot()

    def on_energy_slider(self, val):
        e = slider_to_energy(val)
        self.energy_value_label.setText(f"{e:.1f} keV")

    def get_energy(self):
        return slider_to_energy(self.energy_slider.value())

    def handle_action(self):
        if not self.locked_in:
            self.locked_in = True
            self.elev_slider.setEnabled(False)
            self.azim_slider.setEnabled(False)
            self.energy_slider.setEnabled(False)
            self.reset_btn.setEnabled(False)
            self.action_btn.setText("NEXT TRACK ▶")
            self.action_btn.setStyleSheet(
                "QPushButton { background-color: #00d4e0; color: #000000; "
                "font-size: 16px; font-weight: bold; }"
                "QPushButton:hover { background-color: #4be6f0; }"
            )
            self.calculate_scores()
            self.draw_answers()
        else:
            self.current_track_idx += 1
            if self.current_track_idx < len(self.round_tracks):
                self.load_track()
            else:
                self.end_game()

    def calculate_scores(self):
        row = self.round_tracks[self.current_track_idx]

        elev = np.radians(self.elev_slider.value())
        azim = np.radians(self.azim_slider.value())
        hx = np.cos(elev) * np.cos(azim)
        hy = np.cos(elev) * np.sin(azim)
        hz = np.sin(elev)
        human_v = np.array([hx, hy, hz])

        true_v = np.array([row['true_vx'], row['true_vy'], row['true_vz']])
        ai_v = np.array([row['pred_vx'], row['pred_vy'], row['pred_vz']])

        true_norm = np.linalg.norm(true_v) + 1e-9
        norm_true = true_v / true_norm
        t_elev = np.degrees(np.arcsin(norm_true[2]))
        t_azim = np.degrees(np.arctan2(norm_true[1], norm_true[0]))

        human_dot = np.clip(np.dot(human_v, true_v) / true_norm, -1.0, 1.0)
        ai_dot = np.clip(np.dot(ai_v, true_v) / true_norm, -1.0, 1.0)

        h_error = np.degrees(np.arccos(human_dot))
        ai_error = np.degrees(np.arccos(ai_dot))

        t_energy = row['true_energy']
        h_energy = self.get_energy()
        ai_energy = row['pred_energy']

        h_e_error = abs(h_energy - t_energy)
        ai_e_error = abs(ai_energy - t_energy)

        dir_score = max(0, int(1000 - (h_error * 11.11)))
        ai_dir_score = max(0, int(1000 - (ai_error * 11.11)))
        eng_score = max(0, int(1000 - (h_e_error * 20)))
        ai_eng_score = max(0, int(1000 - (ai_e_error * 20)))

        round_score = dir_score + eng_score
        ai_round_score = ai_dir_score + ai_eng_score

        h_bonus = 0
        if h_error < ai_error:
            h_bonus += 250
        if h_e_error < ai_e_error:
            h_bonus += 250
        round_score += h_bonus

        self.total_score += round_score
        self.ai_total_score += ai_round_score

        # Verdict line
        if round_score > ai_round_score:
            verdict = "<span style='color:#00ff88; font-size:16px; font-weight:bold;'>✅ YOU WON THIS ROUND!</span>"
        elif round_score < ai_round_score:
            verdict = "<span style='color:#ff5555; font-size:16px; font-weight:bold;'>🤖 AI wins this round</span>"
        else:
            verdict = "<span style='color:#ffd700; font-size:16px; font-weight:bold;'>🤝 Tie!</span>"

        feedback = (
            f"{verdict}<br><br>"
            f"<b style='color:#00ff88;'>YOU:</b> "
            f"angle {h_error:.1f}° | energy {h_e_error:.1f} keV | "
            f"<b>+{round_score}</b>"
            f"{' (+' + str(h_bonus) + ' bonus)' if h_bonus > 0 else ''}<br>"
            f"<b style='color:#ff5555;'>AI:</b> "
            f"angle {ai_error:.1f}° | energy {ai_e_error:.1f} keV | "
            f"<b>+{ai_round_score}</b><br><br>"
            f"<b style='color:#a0a0a0;'>Truth:</b> "
            f"E = {t_energy:.1f} keV"
        )
        self.feedback_label.setText(feedback)
        self.update_hud()

    def update_hud(self):
        self.track_label.setText(
            f"Track {self.current_track_idx + 1} / {len(self.round_tracks)}"
        )
        self.human_score_label.setText(f"YOU\n{self.total_score}")
        self.ai_score_label.setText(f"AI\n{self.ai_total_score}")

    def end_game(self):
        self.leaderboard.append({
            "name": self.player_name,
            "score": self.total_score,
            "rounds": len(self.round_tracks),
        })
        self.save_leaderboard()

        # Final score message
        if self.total_score > self.ai_total_score:
            verdict = ("<span style='color:#00ff88;'>🎉  YOU BEAT THE AI!</span><br>"
                       f"You: <b>{self.total_score}</b>  vs  AI: {self.ai_total_score}")
        elif self.total_score < self.ai_total_score:
            verdict = ("<span style='color:#ff5555;'>🤖  AI WINS</span><br>"
                       f"You: {self.total_score}  vs  AI: <b>{self.ai_total_score}</b>")
        else:
            verdict = ("<span style='color:#ffd700;'>🤝  PERFECT TIE!</span><br>"
                       f"Both: <b>{self.total_score}</b>")
        self.final_score_label.setText(verdict)

        # Populate leaderboard
        self.lb_table.setRowCount(len(self.leaderboard))
        for i, entry in enumerate(self.leaderboard):
            rank_item = QTableWidgetItem(f"#{i + 1}")
            rank_item.setTextAlignment(Qt.AlignCenter)
            name_item = QTableWidgetItem(entry['name'])
            name_item.setTextAlignment(Qt.AlignCenter)
            score_item = QTableWidgetItem(str(entry['score']))
            score_item.setTextAlignment(Qt.AlignCenter)
            self.lb_table.setItem(i, 0, rank_item)
            self.lb_table.setItem(i, 1, name_item)
            self.lb_table.setItem(i, 2, score_item)

        self.stacked.setCurrentWidget(self.lb_screen)

    # =====================================================
    # Plotting & Interactions
    # =====================================================
    def on_pick(self, event):
        if self.locked_in:
            return
        try:
            idx = event.ind[0]
            row = self.round_tracks[self.current_track_idx]
            px, py, pz = row['raw_points'][idx]

            r = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
            if r == 0:
                return

            elev = np.degrees(np.arcsin(pz / r))
            azim = np.degrees(np.arctan2(py, px))

            self.elev_slider.blockSignals(True)
            self.azim_slider.blockSignals(True)
            self.elev_slider.setValue(int(elev))
            self.azim_slider.setValue(int(azim))
            self.elev_slider.blockSignals(False)
            self.azim_slider.blockSignals(False)

            self.elev_value_label.setText(f"{int(elev)}°")
            self.azim_value_label.setText(f"{int(azim)}°")

            self.update_plot_live()
        except Exception as e:
            print(f"Click missed: {e}")

    def init_plot(self):
        row = self.round_tracks[self.current_track_idx]
        pts = row['raw_points']

        self.figure.clear()
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#0f1419')
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('#2d3748')
        self.ax.yaxis.pane.set_edgecolor('#2d3748')
        self.ax.zaxis.pane.set_edgecolor('#2d3748')
        self.ax.tick_params(colors='#6b7280', labelsize=9)
        self.ax.xaxis.label.set_color('#a0a0a0')
        self.ax.yaxis.label.set_color('#a0a0a0')
        self.ax.zaxis.label.set_color('#a0a0a0')

        self.ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                        c='#00d4e0', alpha=0.7, s=40,
                        edgecolors='white', linewidths=0.4, picker=8)

        self.ax.set_title(
            f"{len(pts)} Vacancies   |   Click any dot to aim",
            color='white', pad=18, fontsize=14, fontweight='bold')

        if len(pts) > 0:
            max_r = np.max([
                pts[:, 0].max() - pts[:, 0].min(),
                pts[:, 1].max() - pts[:, 1].min(),
                pts[:, 2].max() - pts[:, 2].min(),
            ]) / 2.0
            mx = np.mean(pts[:, 0])
            my = np.mean(pts[:, 1])
            mz = np.mean(pts[:, 2])
            self.ax.set_xlim(mx - max_r, mx + max_r)
            self.ax.set_ylim(my - max_r, my + max_r)
            self.ax.set_zlim(mz - max_r, mz + max_r)

        self.ax.set_xlabel('X (nm)', fontsize=10)
        self.ax.set_ylabel('Y (nm)', fontsize=10)
        self.ax.set_zlabel('Z (nm)', fontsize=10)

        self.figure.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.94)

        self.guess_quiver = None
        self.update_plot_live()

    def update_plot_live(self):
        if self.locked_in:
            return
        if not hasattr(self, 'ax'):
            return

        row = self.round_tracks[self.current_track_idx]
        pts = row['raw_points']
        scale = np.max(np.abs(pts)) if len(pts) > 0 else 10

        elev = np.radians(self.elev_slider.value())
        azim = np.radians(self.azim_slider.value())
        gx = np.cos(elev) * np.cos(azim)
        gy = np.cos(elev) * np.sin(azim)
        gz = np.sin(elev)

        if self.guess_quiver is not None:
            self.guess_quiver.remove()

        self.guess_quiver = self.ax.quiver(
            0, 0, 0, gx * scale, gy * scale, gz * scale,
            color='#ffd700', linewidth=6,
            arrow_length_ratio=0.18,
            label='Your Guess'
        )

        self.canvas.draw_idle()

    def draw_answers(self):
        row = self.round_tracks[self.current_track_idx]
        pts = row['raw_points']
        scale = np.max(np.abs(pts)) if len(pts) > 0 else 10

        self.ax.quiver(
            0, 0, 0,
            row['true_vx'] * scale, row['true_vy'] * scale, row['true_vz'] * scale,
            color='#00ff88', linewidth=4,
            arrow_length_ratio=0.18,
            label='Truth'
        )
        self.ax.quiver(
            0, 0, 0,
            row['pred_vx'] * scale, row['pred_vy'] * scale, row['pred_vz'] * scale,
            color='#ff5555', linewidth=4,
            arrow_length_ratio=0.18,
            label='AI Guess'
        )

        self.ax.set_title(
            f"True Energy: {row['true_energy']:.1f} keV   |   {len(pts)} Vacancies",
            color='white', pad=18, fontsize=14, fontweight='bold')
        legend = self.ax.legend(
            facecolor='#1a1f24', edgecolor='#2d3748',
            labelcolor='white', fontsize=12, loc='upper right')
        legend.get_frame().set_alpha(0.95)

        self.canvas.draw_idle()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Better DPI scaling on Mac retina displays
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    window = VectorGame()
    window.show()
    sys.exit(app.exec_())
