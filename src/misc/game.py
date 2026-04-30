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

# Filter: only show tracks in this energy range so the game is a fair fight.
# (1 keV tracks are bimodal/ambiguous and frustrating; >100 keV is rare.)
MIN_ENERGY_KEV = 5.0
MAX_ENERGY_KEV = 100.0

# ===============================
# Theme
# ===============================
# Refined neutral palette inspired by Linear/Notion/Stripe.
# One muted accent (slate-blue), restrained borders, no glow effects.
MODERN_THEME = """
QWidget {
    background-color: #0d0d0f;
    color: #d4d4d6;
    font-family: -apple-system, 'SF Pro Text', 'Inter', 'Segoe UI', sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #232328;
    border-radius: 6px;
    margin-top: 14px;
    font-weight: 500;
    font-size: 11px;
    color: #8b8b94;
    padding-top: 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 6px;
    background-color: #0d0d0f;
    text-transform: uppercase;
    letter-spacing: 1px;
}
QPushButton {
    background-color: #18181b;
    border: 1px solid #27272a;
    border-radius: 6px;
    padding: 9px 14px;
    font-weight: 500;
    font-size: 13px;
    color: #d4d4d6;
}
QPushButton:hover {
    background-color: #1f1f23;
    border: 1px solid #3f3f46;
    color: #ffffff;
}
QPushButton:pressed {
    background-color: #27272a;
}
QPushButton:disabled {
    background-color: #131316;
    color: #4a4a50;
    border: 1px solid #1d1d20;
}
QLineEdit, QSpinBox {
    background-color: #131316;
    border: 1px solid #27272a;
    padding: 8px 10px;
    border-radius: 6px;
    color: #ffffff;
    font-size: 13px;
    selection-background-color: #3b3b40;
}
QLineEdit:focus, QSpinBox:focus {
    border: 1px solid #6366f1;
}
QSlider::groove:horizontal {
    border: none;
    height: 4px;
    background: #232328;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #d4d4d6;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
    border: none;
}
QSlider::handle:horizontal:hover {
    background: #ffffff;
}
QSlider::sub-page:horizontal {
    background: #6366f1;
    border-radius: 2px;
}
QTableWidget {
    background-color: #131316;
    border: 1px solid #232328;
    border-radius: 6px;
    gridline-color: #1d1d20;
    font-size: 13px;
}
QTableWidget::item {
    padding: 8px;
}
QTableWidget::item:selected {
    background-color: #1f1f23;
    color: #ffffff;
}
QHeaderView::section {
    background-color: #0d0d0f;
    padding: 10px;
    border: none;
    border-bottom: 1px solid #232328;
    font-weight: 500;
    font-size: 11px;
    color: #8b8b94;
    text-transform: uppercase;
    letter-spacing: 1px;
}
QLabel { color: #d4d4d6; }
"""




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
        self.human_wins = 0
        self.ai_wins = 0
        self.human_error_sum = 0.0
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
        # Sort by win-rate first, then by lowest avg error as tiebreaker.
        # Migrate any legacy 'score' entries (best-effort).
        for entry in self.leaderboard:
            if 'wins' not in entry:
                entry['wins'] = 0
            if 'rounds' not in entry:
                entry['rounds'] = 1
            if 'avg_error' not in entry:
                entry['avg_error'] = 999.0

        self.leaderboard.sort(
            key=lambda x: (
                -(x['wins'] / max(x['rounds'], 1)),  # higher win rate first
                x['avg_error'],                       # then lower error
            )
        )
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
        layout.setSpacing(28)

        eyebrow = QLabel("Direction Reconstruction Challenge")
        eyebrow.setStyleSheet("font-size: 11px; color: #6366f1; "
                              "letter-spacing: 2px; text-transform: uppercase; "
                              "font-weight: 500; margin-top: 60px;")
        eyebrow.setAlignment(Qt.AlignCenter)
        layout.addWidget(eyebrow)

        title = QLabel("Human vs. Model")
        title.setStyleSheet("font-size: 44px; font-weight: 600; color: #ffffff; "
                            "letter-spacing: -1px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        desc = QLabel(
            "Inspect a 3D point cloud of vacancies left by a nuclear recoil. "
            "Estimate the direction the particle came from. "
            "Compare your guess against a trained graph neural network."
        )
        desc.setStyleSheet("font-size: 14px; color: #8b8b94; line-height: 1.6;")
        desc.setAlignment(Qt.AlignCenter)
        desc.setWordWrap(True)
        desc.setFixedWidth(540)
        layout.addWidget(desc, alignment=Qt.AlignCenter)

        # Form
        form_widget = QFrame()
        form_widget.setStyleSheet("QFrame { background-color: transparent; }")
        form_widget.setFixedWidth(360)
        form = QFormLayout(form_widget)
        form.setSpacing(14)
        form.setContentsMargins(0, 12, 0, 0)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Your name")
        self.name_input.setFixedHeight(38)

        self.rounds_input = QSpinBox()
        self.rounds_input.setRange(1, 20)
        self.rounds_input.setValue(5)
        self.rounds_input.setFixedHeight(38)

        name_lbl = QLabel("Name")
        name_lbl.setStyleSheet("font-size: 13px; color: #8b8b94;")
        rounds_lbl = QLabel("Tracks")
        rounds_lbl.setStyleSheet("font-size: 13px; color: #8b8b94;")

        form.addRow(name_lbl, self.name_input)
        form.addRow(rounds_lbl, self.rounds_input)

        layout.addWidget(form_widget, alignment=Qt.AlignCenter)

        start_btn = QPushButton("Begin")
        start_btn.setFixedSize(200, 42)
        start_btn.setStyleSheet(
            "QPushButton { background-color: #6366f1; color: #ffffff; "
            "font-size: 14px; font-weight: 500; border-radius: 6px; "
            "border: none; }"
            "QPushButton:hover { background-color: #5558e3; }"
            "QPushButton:pressed { background-color: #4f51d9; }"
        )
        start_btn.clicked.connect(self.start_game)
        layout.addWidget(start_btn, alignment=Qt.AlignCenter)

    # =====================================================
    # SCREEN 2: GAMEPLAY
    # =====================================================
    def build_game_screen(self):
        self.game_screen = QWidget()
        outer = QHBoxLayout(self.game_screen)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(20)

        # ── Left Panel (Controls) ──
        left_panel = QWidget()
        left_panel.setFixedWidth(380)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(20)
        left_layout.setContentsMargins(4, 4, 4, 4)

        # HUD: simple track counter + inline scores
        hud_frame = QFrame()
        hud_layout = QVBoxLayout(hud_frame)
        hud_layout.setSpacing(14)
        hud_layout.setContentsMargins(0, 0, 0, 0)

        self.track_label = QLabel("Track 1 of 5")
        self.track_label.setStyleSheet("font-size: 12px; color: #6366f1; "
                                       "letter-spacing: 1px; text-transform: uppercase; "
                                       "font-weight: 500;")
        hud_layout.addWidget(self.track_label)

        score_row = QHBoxLayout()
        score_row.setSpacing(20)

        you_col = QVBoxLayout()
        you_col.setSpacing(2)
        you_label = QLabel("Your wins")
        you_label.setStyleSheet("font-size: 11px; color: #6b6b72; "
                                "text-transform: uppercase; letter-spacing: 1px;")
        self.human_score_label = QLabel("0")
        self.human_score_label.setStyleSheet("font-size: 28px; color: #ffffff; "
                                             "font-weight: 600;")
        you_col.addWidget(you_label)
        you_col.addWidget(self.human_score_label)
        score_row.addLayout(you_col)

        ai_col = QVBoxLayout()
        ai_col.setSpacing(2)
        ai_label = QLabel("Model wins")
        ai_label.setStyleSheet("font-size: 11px; color: #6b6b72; "
                               "text-transform: uppercase; letter-spacing: 1px;")
        self.ai_score_label = QLabel("0")
        self.ai_score_label.setStyleSheet("font-size: 28px; color: #8b8b94; "
                                          "font-weight: 600;")
        ai_col.addWidget(ai_label)
        ai_col.addWidget(self.ai_score_label)
        score_row.addLayout(ai_col)

        score_row.addStretch()
        hud_layout.addLayout(score_row)

        # Subtle divider
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet("color: #232328; background-color: #232328; "
                              "max-height: 1px;")
        hud_layout.addWidget(divider)

        left_layout.addWidget(hud_frame)

        # Direction controls
        dir_group = QGroupBox("Direction")
        dir_layout = QVBoxLayout()
        dir_layout.setSpacing(12)
        dir_layout.setContentsMargins(0, 8, 0, 4)

        elev_row = QHBoxLayout()
        elev_lbl = QLabel("Elevation")
        elev_lbl.setStyleSheet("font-size: 13px; color: #8b8b94;")
        self.elev_value_label = QLabel("0°")
        self.elev_value_label.setStyleSheet("font-size: 13px; color: #ffffff; "
                                            "font-weight: 500;")
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
        azim_lbl = QLabel("Azimuth")
        azim_lbl.setStyleSheet("font-size: 13px; color: #8b8b94;")
        self.azim_value_label = QLabel("0°")
        self.azim_value_label.setStyleSheet("font-size: 13px; color: #ffffff; "
                                            "font-weight: 500;")
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

        # Subtle inline hint
        hint = QLabel("Or click a vacancy to aim")
        hint.setStyleSheet("font-size: 12px; color: #5a5a60; font-style: italic; "
                           "padding-top: 4px;")
        dir_layout.addWidget(hint)

        dir_group.setLayout(dir_layout)
        left_layout.addWidget(dir_group)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setFixedHeight(40)
        self.reset_btn.clicked.connect(self.reset_aim)
        btn_row.addWidget(self.reset_btn, stretch=1)

        self.action_btn = QPushButton("Submit")
        self.action_btn.setFixedHeight(40)
        self.action_btn.setStyleSheet(
            "QPushButton { background-color: #6366f1; color: #ffffff; "
            "border: none; font-weight: 500; }"
            "QPushButton:hover { background-color: #5558e3; }"
            "QPushButton:pressed { background-color: #4f51d9; }"
        )
        self.action_btn.clicked.connect(self.handle_action)
        btn_row.addWidget(self.action_btn, stretch=2)

        left_layout.addLayout(btn_row)

        # Feedback panel
        self.feedback_label = QLabel("Aim your direction and submit when ready.")
        self.feedback_label.setWordWrap(True)
        self.feedback_label.setStyleSheet(
            "font-size: 13px; padding: 14px; background-color: #131316; "
            "border-radius: 6px; border: 1px solid #232328; color: #8b8b94; "
            "line-height: 1.5;"
        )
        self.feedback_label.setMinimumHeight(160)
        left_layout.addWidget(self.feedback_label)

        # Help link at bottom
        help_row = QHBoxLayout()
        self.info_btn = QPushButton("How to play")
        self.info_btn.setFixedHeight(28)
        self.info_btn.setStyleSheet(
            "QPushButton { background-color: transparent; color: #6b6b72; "
            "font-size: 12px; border: none; padding: 4px; text-align: left; }"
            "QPushButton:hover { color: #ffffff; background-color: transparent; "
            "border: none; }"
        )
        self.info_btn.clicked.connect(self.show_instructions)
        help_row.addWidget(self.info_btn)
        help_row.addStretch()
        left_layout.addLayout(help_row)

        left_layout.addStretch()
        outer.addWidget(left_panel)

        # ── Right Panel (3D Plot) ──
        plot_container = QFrame()
        plot_container.setStyleSheet(
            "QFrame { background-color: #131316; border-radius: 8px; "
            "border: 1px solid #232328; }"
        )
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(8, 8, 8, 8)

        self.figure = Figure(facecolor='#131316')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('pick_event', self.on_pick)
        plot_layout.addWidget(self.canvas)

        outer.addWidget(plot_container, stretch=1)

    # =====================================================
    # SCREEN 3: LEADERBOARD
    # =====================================================
    def build_leaderboard_screen(self):
        self.lb_screen = QWidget()
        layout = QVBoxLayout(self.lb_screen)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)

        eyebrow = QLabel("Final result")
        eyebrow.setStyleSheet("font-size: 11px; color: #6366f1; "
                              "letter-spacing: 2px; text-transform: uppercase; "
                              "font-weight: 500; margin-top: 60px;")
        eyebrow.setAlignment(Qt.AlignCenter)
        layout.addWidget(eyebrow)

        self.final_score_label = QLabel("")
        self.final_score_label.setStyleSheet(
            "font-size: 18px; color: #d4d4d6; line-height: 1.6;")
        self.final_score_label.setAlignment(Qt.AlignCenter)
        self.final_score_label.setFixedWidth(500)
        layout.addWidget(self.final_score_label, alignment=Qt.AlignCenter)

        lb_title = QLabel("Leaderboard")
        lb_title.setStyleSheet("font-size: 11px; color: #6b6b72; "
                               "letter-spacing: 2px; text-transform: uppercase; "
                               "font-weight: 500; margin-top: 24px;")
        lb_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(lb_title)

        self.lb_table = QTableWidget()
        self.lb_table.setColumnCount(4)
        self.lb_table.setHorizontalHeaderLabels(
            ["Rank", "Player", "Wins", "Avg error"])
        self.lb_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.lb_table.setFixedWidth(520)
        self.lb_table.setFixedHeight(340)
        self.lb_table.verticalHeader().setVisible(False)
        self.lb_table.setShowGrid(False)
        layout.addWidget(self.lb_table, alignment=Qt.AlignCenter)

        replay_btn = QPushButton("Play again")
        replay_btn.setFixedSize(200, 42)
        replay_btn.setStyleSheet(
            "QPushButton { background-color: #6366f1; color: #ffffff; "
            "font-size: 14px; font-weight: 500; border-radius: 6px; "
            "border: none; }"
            "QPushButton:hover { background-color: #5558e3; }"
            "QPushButton:pressed { background-color: #4f51d9; }"
        )
        replay_btn.clicked.connect(lambda: self.stacked.setCurrentWidget(self.start_screen))
        layout.addWidget(replay_btn, alignment=Qt.AlignCenter)

    # =====================================================
    # Game Logic
    # =====================================================
    def show_instructions(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("How to play")
        msg.setStyleSheet(
            "QLabel { color: #d4d4d6; min-width: 460px; font-size: 13px; "
            "line-height: 1.6; }"
            "QMessageBox { background-color: #131316; }"
            "QPushButton { background-color: #6366f1; color: #ffffff; "
            "border-radius: 6px; padding: 8px 20px; font-weight: 500; "
            "border: none; }"
            "QPushButton:hover { background-color: #5558e3; }"
        )
        msg.setText(
            "<p style='color:#ffffff; font-size:15px; font-weight:600; "
            "margin-bottom:12px;'>How to play</p>"
            "<p><span style='color:#6366f1;'>1.</span> "
            "<b style='color:#ffffff;'>Aim your arrow.</b> "
            "Click any vacancy dot on the 3D plot to point your arrow at it, "
            "or use the elevation/azimuth sliders for fine adjustments.</p>"
            "<p><span style='color:#6366f1;'>2.</span> "
            "<b style='color:#ffffff;'>Submit.</b> "
            "Reveals your guess, the model's prediction, and the ground truth.</p>"
            "<p style='color:#8b8b94; margin-top:14px; font-size:12px;'>"
            "You win a round if your angle is closer to the truth than the model's. "
            "Final score is the number of rounds won.</p>"
        )
        msg.exec_()

    def start_game(self):
        self.player_name = self.name_input.text().strip() or "Guest"
        num_rounds = self.rounds_input.value()

        self.round_tracks = self.df_results.sample(n=num_rounds).to_dict('records')
        self.current_track_idx = 0
        self.human_wins = 0
        self.ai_wins = 0
        self.human_error_sum = 0.0
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
        self.reset_btn.setEnabled(True)

        self.elev_slider.blockSignals(True)
        self.azim_slider.blockSignals(True)
        self.elev_slider.setValue(0)
        self.azim_slider.setValue(0)
        self.elev_slider.blockSignals(False)
        self.azim_slider.blockSignals(False)

        self.elev_value_label.setText("0°")
        self.azim_value_label.setText("0°")

        self.action_btn.setText("Submit")
        self.action_btn.setStyleSheet(
            "QPushButton { background-color: #6366f1; color: #ffffff; "
            "border: none; font-weight: 500; }"
            "QPushButton:hover { background-color: #5558e3; }"
            "QPushButton:pressed { background-color: #4f51d9; }"
        )
        self.feedback_label.setText("Aim your direction and submit when ready.")
        self.update_hud()
        self.init_plot()

    def handle_action(self):
        if not self.locked_in:
            self.locked_in = True
            self.elev_slider.setEnabled(False)
            self.azim_slider.setEnabled(False)
            self.reset_btn.setEnabled(False)
            self.action_btn.setText("Next track")
            self.action_btn.setStyleSheet(
                "QPushButton { background-color: #ffffff; color: #0d0d0f; "
                "border: none; font-weight: 500; }"
                "QPushButton:hover { background-color: #e5e5e7; }"
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

        human_dot = np.clip(np.dot(human_v, true_v) / true_norm, -1.0, 1.0)
        ai_dot = np.clip(np.dot(ai_v, true_v) / true_norm, -1.0, 1.0)

        h_error = np.degrees(np.arccos(human_dot))
        ai_error = np.degrees(np.arccos(ai_dot))

        self.human_error_sum += h_error

        if h_error < ai_error:
            self.human_wins += 1
            verdict = "<span style='color:#22c55e; font-weight:600;'>You won this round.</span>"
        elif ai_error < h_error:
            self.ai_wins += 1
            verdict = "<span style='color:#8b8b94; font-weight:600;'>Model won this round.</span>"
        else:
            verdict = "<span style='color:#a78bfa; font-weight:600;'>Tie.</span>"

        feedback = (
            f"{verdict}<br><br>"
            f"<span style='color:#6b6b72;'>You</span>"
            f"<span style='color:#ffffff;'>  {h_error:.1f}°</span><br>"
            f"<span style='color:#6b6b72;'>Model</span>"
            f"<span style='color:#ffffff;'>  {ai_error:.1f}°</span>"
        )
        self.feedback_label.setText(feedback)
        self.update_hud()

    def update_hud(self):
        self.track_label.setText(
            f"Track {self.current_track_idx + 1} of {len(self.round_tracks)}"
        )
        self.human_score_label.setText(str(self.human_wins))
        self.ai_score_label.setText(str(self.ai_wins))

    def end_game(self):
        n_rounds = len(self.round_tracks)
        avg_err = self.human_error_sum / max(n_rounds, 1)

        self.leaderboard.append({
            "name": self.player_name,
            "wins": self.human_wins,
            "rounds": n_rounds,
            "avg_error": round(avg_err, 2),
        })
        self.save_leaderboard()

        if self.human_wins > self.ai_wins:
            headline = "You beat the model."
        elif self.human_wins < self.ai_wins:
            headline = "Model wins."
        else:
            headline = "Tie."

        verdict = (
            f"<span style='color:#ffffff; font-size:32px; "
            f"font-weight:600;'>{headline}</span><br><br>"
            f"<span style='color:#8b8b94;'>"
            f"You won {self.human_wins} of {n_rounds}  ·  "
            f"Avg error {avg_err:.1f}°</span>"
        )
        self.final_score_label.setText(verdict)

        # Populate leaderboard
        self.lb_table.setRowCount(len(self.leaderboard))
        for i, entry in enumerate(self.leaderboard):
            rank_item = QTableWidgetItem(f"#{i + 1}")
            rank_item.setTextAlignment(Qt.AlignCenter)
            name_item = QTableWidgetItem(entry['name'])
            name_item.setTextAlignment(Qt.AlignCenter)
            wins_item = QTableWidgetItem(
                f"{entry['wins']} / {entry['rounds']}")
            wins_item.setTextAlignment(Qt.AlignCenter)
            err_item = QTableWidgetItem(f"{entry['avg_error']:.1f}°")
            err_item.setTextAlignment(Qt.AlignCenter)
            self.lb_table.setItem(i, 0, rank_item)
            self.lb_table.setItem(i, 1, name_item)
            self.lb_table.setItem(i, 2, wins_item)
            self.lb_table.setItem(i, 3, err_item)

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
        self.ax.set_facecolor('#131316')
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('#1f1f23')
        self.ax.yaxis.pane.set_edgecolor('#1f1f23')
        self.ax.zaxis.pane.set_edgecolor('#1f1f23')
        self.ax.tick_params(colors='#5a5a60', labelsize=8)
        self.ax.xaxis.label.set_color('#6b6b72')
        self.ax.yaxis.label.set_color('#6b6b72')
        self.ax.zaxis.label.set_color('#6b6b72')

        self.ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                        c='#a5b4fc', alpha=0.65, s=38,
                        edgecolors='#0d0d0f', linewidths=0.3, picker=8)

        self.ax.set_title(
            f"{len(pts)} vacancies",
            color='#d4d4d6', pad=14, fontsize=13, fontweight='normal')

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
            color='#ffffff', linewidth=4,
            arrow_length_ratio=0.18,
            label='You'
        )

        self.canvas.draw_idle()

    def draw_answers(self):
        row = self.round_tracks[self.current_track_idx]
        pts = row['raw_points']
        scale = np.max(np.abs(pts)) if len(pts) > 0 else 10

        self.ax.quiver(
            0, 0, 0,
            row['true_vx'] * scale, row['true_vy'] * scale, row['true_vz'] * scale,
            color='#22c55e', linewidth=3.5,
            arrow_length_ratio=0.18,
            label='Truth'
        )
        self.ax.quiver(
            0, 0, 0,
            row['pred_vx'] * scale, row['pred_vy'] * scale, row['pred_vz'] * scale,
            color='#a78bfa', linewidth=3.5,
            arrow_length_ratio=0.18,
            label='Model'
        )

        self.ax.set_title(
            f"E = {row['true_energy']:.1f} keV  ·  {len(pts)} vacancies",
            color='#d4d4d6', pad=14, fontsize=13, fontweight='normal')
        legend = self.ax.legend(
            facecolor='#131316', edgecolor='#232328',
            labelcolor='#d4d4d6', fontsize=11, loc='upper right')
        legend.get_frame().set_alpha(0.95)

        self.canvas.draw_idle()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Better DPI scaling on Mac retina displays
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    window = VectorGame()
    window.show()
    sys.exit(app.exec_())
