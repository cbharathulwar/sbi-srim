import sys
import os
import json
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox, QSlider, QStackedWidget,
    QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox, QFormLayout,
    QMessageBox  # <-- Added for the instructions popup
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

plt.style.use('dark_background')

# ===============================
# Config & Paths
# ===============================
RESULTS_CSV = "results/pipeline_b/pointnet_3d_eval_results.csv"
RAW_DATA_CSV = "data/mcpe-3d/mcpe_3d_eval_10k.csv" 
LEADERBOARD_FILE = "results/pipeline_b/leaderboard.json"

MODERN_THEME = """
QWidget { background-color: #1e1e1e; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; font-size: 14px; }
QGroupBox { border: 1px solid #3a3a3a; border-radius: 6px; margin-top: 15px; font-weight: bold; color: #00adb5; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 10px; padding: 0 5px; }
QPushButton { background-color: #2d2d2d; border: 1px solid #3a3a3a; border-radius: 5px; padding: 10px; font-weight: bold; }
QPushButton:hover { background-color: #3d3d3d; border: 1px solid #00adb5; color: #ffffff; }
QPushButton:pressed { background-color: #00adb5; color: #000000; }
QLineEdit, QSpinBox, QDoubleSpinBox { background-color: #2d2d2d; border: 1px solid #3a3a3a; padding: 5px; border-radius: 4px; color: white; }
QSlider::groove:horizontal { border: 1px solid #3a3a3a; height: 8px; background: #2d2d2d; border-radius: 4px; }
QSlider::handle:horizontal { background: #00adb5; width: 16px; margin: -4px 0; border-radius: 8px; }
QTableWidget { background-color: #2d2d2d; border: none; gridline-color: #3a3a3a; }
QHeaderView::section { background-color: #1e1e1e; padding: 4px; border: 1px solid #3a3a3a; font-weight: bold; }
"""

# ===============================
# Main Application
# ===============================
class VectorGame(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Human vs. AI: 3D Track Challenge")
        self.resize(1300, 900)
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
        print("[INFO] Loading Game Data...")
        self.df_results = pd.read_csv(RESULTS_CSV)
        df_raw = pd.read_csv(RAW_DATA_CSV)
        
        # Center points and attach
        grouped = df_raw.groupby('ion_number')
        pts_list = []
        for _, group in grouped:
            pts = group[['x', 'y', 'z']].values
            pts = pts - np.mean(pts, axis=0)
            pts_list.append(pts)
        self.df_results['raw_points'] = pts_list

    def load_leaderboard(self):
        if os.path.exists(LEADERBOARD_FILE):
            with open(LEADERBOARD_FILE, 'r') as f:
                self.leaderboard = json.load(f)
        else:
            self.leaderboard = []

    def save_leaderboard(self):
        self.leaderboard.sort(key=lambda x: x['score'], reverse=True)
        self.leaderboard = self.leaderboard[:10]  # Keep Top 10
        with open(LEADERBOARD_FILE, 'w') as f:
            json.dump(self.leaderboard, f)

    # --- SCREEN 1: START MENU ---
    def build_start_screen(self):
        self.start_screen = QWidget()
        layout = QVBoxLayout(self.start_screen)
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("MCPE 3D: Human vs. AI")
        title.setStyleSheet("font-size: 36px; font-weight: bold; color: #00adb5; margin-bottom: 20px;")
        layout.addWidget(title, alignment=Qt.AlignCenter)

        form = QFormLayout()
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter your name...")
        self.name_input.setFixedWidth(200)
        
        self.rounds_input = QSpinBox()
        self.rounds_input.setRange(1, 100)
        self.rounds_input.setValue(5)
        self.rounds_input.setFixedWidth(200)

        form.addRow("Player Name:", self.name_input)
        form.addRow("Tracks per Game:", self.rounds_input)
        
        form_widget = QWidget()
        form_widget.setLayout(form)
        layout.addWidget(form_widget, alignment=Qt.AlignCenter)

        start_btn = QPushButton("Start Challenge")
        start_btn.setFixedWidth(200)
        start_btn.setStyleSheet("font-size: 16px; margin-top: 20px; background-color: #00adb5; color: black;")
        start_btn.clicked.connect(self.start_game)
        layout.addWidget(start_btn, alignment=Qt.AlignCenter)

    # --- SCREEN 2: GAMEPLAY ---
    def build_game_screen(self):
        self.game_screen = QWidget()
        layout = QHBoxLayout(self.game_screen)
        
        # Left Panel (Controls)
        left_panel = QWidget()
        left_panel.setFixedWidth(380)
        left_layout = QVBoxLayout(left_panel)

        # HUD + Info Button Layout
        hud_layout = QHBoxLayout()
        self.hud_label = QLabel("Track 1 / 5\nScore: 0")
        self.hud_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px; background: #2d2d2d; border-radius: 5px;")
        
        self.info_btn = QPushButton("❓")
        self.info_btn.setFixedSize(40, 40)
        self.info_btn.setStyleSheet("border-radius: 20px; background-color: #3a3a3a; font-size: 18px; color: #00adb5;")
        self.info_btn.clicked.connect(self.show_instructions)
        
        hud_layout.addWidget(self.hud_label, stretch=1)
        hud_layout.addWidget(self.info_btn)
        left_layout.addLayout(hud_layout)

        # Vector Controls 
        ctrl_group = QGroupBox("1. Aim Your Vector")
        ctrl_layout = QVBoxLayout()
        
        hint_label = QLabel("🎯 Hint: Click any dot on the 3D plot to auto-aim!")
        hint_label.setStyleSheet("color: #ffcc00; font-weight: bold; margin-bottom: 5px;")
        ctrl_layout.addWidget(hint_label)

        ctrl_layout.addWidget(QLabel("Elevation (Fine-tune Up/Down)"))
        self.elev_slider = QSlider(Qt.Horizontal)
        self.elev_slider.setRange(-90, 90)
        self.elev_slider.valueChanged.connect(self.update_plot_live)
        ctrl_layout.addWidget(self.elev_slider)

        ctrl_layout.addWidget(QLabel("Azimuth (Fine-tune Left/Right)"))
        self.azim_slider = QSlider(Qt.Horizontal)
        self.azim_slider.setRange(-180, 180)
        self.azim_slider.valueChanged.connect(self.update_plot_live)
        ctrl_layout.addWidget(self.azim_slider)

        ctrl_group.setLayout(ctrl_layout)
        left_layout.addWidget(ctrl_group)

        # Energy Input 
        eng_group = QGroupBox("2. Guess The Energy")
        eng_layout = QVBoxLayout()
        self.energy_spinbox = QDoubleSpinBox()
        self.energy_spinbox.setRange(0.0, 500.0)
        self.energy_spinbox.setDecimals(1)
        self.energy_spinbox.setValue(10.0)
        self.energy_spinbox.setSuffix(" keV")
        eng_layout.addWidget(QLabel("Based on the number of vacancies, what is the energy?"))
        eng_layout.addWidget(self.energy_spinbox)
        eng_group.setLayout(eng_layout)
        left_layout.addWidget(eng_group)

        self.action_btn = QPushButton("Lock In Guess")
        self.action_btn.setStyleSheet("background-color: #ff3333; color: white; font-size: 16px; margin-top: 10px;")
        self.action_btn.clicked.connect(self.handle_action)
        left_layout.addWidget(self.action_btn)

        self.feedback_label = QLabel("")
        self.feedback_label.setWordWrap(True)
        self.feedback_label.setStyleSheet("font-size: 13px; padding: 10px; background-color: #222222; border-radius: 5px; margin-top: 10px;")
        left_layout.addWidget(self.feedback_label)

        left_layout.addStretch()
        layout.addWidget(left_panel)

        # Right Panel (Plot)
        self.figure = Figure(facecolor='#1e1e1e')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('pick_event', self.on_pick)
        layout.addWidget(self.canvas, stretch=1)

    # --- SCREEN 3: LEADERBOARD ---
    def build_leaderboard_screen(self):
        self.lb_screen = QWidget()
        layout = QVBoxLayout(self.lb_screen)
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("Global Leaderboard")
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: #00adb5;")
        layout.addWidget(title, alignment=Qt.AlignCenter)

        self.lb_table = QTableWidget()
        self.lb_table.setColumnCount(3)
        self.lb_table.setHorizontalHeaderLabels(["Rank", "Player", "Score"])
        self.lb_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.lb_table.setFixedWidth(500)
        layout.addWidget(self.lb_table, alignment=Qt.AlignCenter)

        replay_btn = QPushButton("Play Again")
        replay_btn.setFixedWidth(200)
        replay_btn.clicked.connect(lambda: self.stacked.setCurrentWidget(self.start_screen))
        layout.addWidget(replay_btn, alignment=Qt.AlignCenter)

    # ===============================
    # Game Logic
    # ===============================
    def show_instructions(self):
        """Displays a popup with instructions and scoring rules."""
        msg = QMessageBox(self)
        msg.setWindowTitle("How to Play & Scoring")
        msg.setStyleSheet("QLabel { color: white; min-width: 400px; } QMessageBox { background-color: #2d2d2d; } QPushButton { background-color: #00adb5; color: black; border-radius: 5px; padding: 5px 15px; font-weight: bold; }")
        msg.setText(
            "<h3>🎯 How to Play</h3>"
            "<b>1. Aim Your Vector:</b> Drag the 3D plot to look around. Click directly on any blue vacancy dot to automatically snap your yellow guessing arrow to it. Use the sliders for micro-adjustments.<br><br>"
            "<b>2. Guess the Energy:</b> Look at the total number of vacancies listed at the top of the plot. Remember: higher energy particles create more vacancies!<br><br>"
            "<h3>🏆 Scoring (2000 Base + 500 Bonus)</h3>"
            "• <b>Direction (1000 pts):</b> You lose ~11 points for every degree your angle is off.<br>"
            "• <b>Energy (1000 pts):</b> You lose 20 points for every 1 keV you are off.<br>"
            "• <b>Bonus:</b> Beat the AI's angle? +250 points. Beat the AI's energy? +250 points.<br>"
        )
        msg.exec_()

    def start_game(self):
        self.player_name = self.name_input.text().strip() or "Guest"
        num_rounds = self.rounds_input.value()
        
        # Pick random tracks
        self.round_tracks = self.df_results.sample(n=num_rounds).to_dict('records')
        self.current_track_idx = 0
        self.total_score = 0
        self.ai_total_score = 0
        
        self.stacked.setCurrentWidget(self.game_screen)
        self.load_track()

    def load_track(self):
        self.locked_in = False
        self.elev_slider.setEnabled(True)
        self.azim_slider.setEnabled(True)
        self.energy_spinbox.setEnabled(True)
        
        # Block signals briefly so resetting the sliders doesn't trigger a redraw 
        # before the plot is even built!
        self.elev_slider.blockSignals(True)
        self.azim_slider.blockSignals(True)
        self.elev_slider.setValue(0)
        self.azim_slider.setValue(0)
        self.elev_slider.blockSignals(False)
        self.azim_slider.blockSignals(False)
        
        self.energy_spinbox.setValue(10.0)
        
        self.action_btn.setText("Lock In Guess")
        self.action_btn.setStyleSheet("background-color: #ff3333; color: white;")
        self.feedback_label.setText("Analyze the track and lock in your guess.")
        self.update_hud()
        
        # Build the static plot elements (vacancies, bounds, etc.)
        self.init_plot()

    def handle_action(self):
        if not self.locked_in:
            self.locked_in = True
            self.elev_slider.setEnabled(False)
            self.azim_slider.setEnabled(False)
            self.energy_spinbox.setEnabled(False)
            self.action_btn.setText("Next Track ▶")
            self.action_btn.setStyleSheet("background-color: #00adb5; color: black;")
            self.calculate_scores()
            # Draw the true vectors directly onto the existing plot
            self.draw_answers() 
        else:
            self.current_track_idx += 1
            if self.current_track_idx < len(self.round_tracks):
                self.load_track()
            else:
                self.end_game()


    def calculate_scores(self):
        row = self.round_tracks[self.current_track_idx]
        
        # --- 1. DIRECTION MATH ---
        elev = np.radians(self.elev_slider.value())
        azim = np.radians(self.azim_slider.value())
        hx = np.cos(elev) * np.cos(azim)
        hy = np.cos(elev) * np.sin(azim)
        hz = np.sin(elev)
        human_v = np.array([hx, hy, hz])

        true_v = np.array([row['true_vx'], row['true_vy'], row['true_vz']])
        ai_v = np.array([row['pred_vx'], row['pred_vy'], row['pred_vz']])

        # Calculate True Elevation/Azimuth for Display
        true_norm = np.linalg.norm(true_v) + 1e-9
        norm_true = true_v / true_norm
        t_elev = np.degrees(np.arcsin(norm_true[2]))
        t_azim = np.degrees(np.arctan2(norm_true[1], norm_true[0]))

        human_dot = np.clip(np.dot(human_v, true_v) / true_norm, -1.0, 1.0)
        ai_dot = np.clip(np.dot(ai_v, true_v) / true_norm, -1.0, 1.0)
        
        h_error = np.degrees(np.arccos(human_dot))
        ai_error = np.degrees(np.arccos(ai_dot))

        # --- 2. ENERGY MATH ---
        t_energy = row['true_energy']
        h_energy = self.energy_spinbox.value()
        ai_energy = row['pred_energy']
        
        h_e_error = abs(h_energy - t_energy)
        ai_e_error = abs(ai_energy - t_energy)

        # --- SCORING ---
        dir_score = max(0, int(1000 - (h_error * 11.11)))
        ai_dir_score = max(0, int(1000 - (ai_error * 11.11)))
        
        eng_score = max(0, int(1000 - (h_e_error * 20)))
        ai_eng_score = max(0, int(1000 - (ai_e_error * 20)))
        
        round_score = dir_score + eng_score
        ai_round_score = ai_dir_score + ai_eng_score
        
        # BONUSES
        h_bonus = 0
        if h_error < ai_error: h_bonus += 250
        if h_e_error < ai_e_error: h_bonus += 250
        round_score += h_bonus

        self.total_score += round_score
        self.ai_total_score += ai_round_score
        
        # The new feedback string with Ground Truth included
        feedback = (
            f"<b style='color:#00ff00; font-size:14px;'>✓ GROUND TRUTH:</b><br>"
            f"Elevation: {t_elev:.1f}° | Azimuth: {t_azim:.1f}°<br>"
            f"Energy: {t_energy:.1f} keV<br><hr style='border: 1px solid #3a3a3a;'>"
            f"<b style='color:#00adb5;'>HUMAN:</b><br>"
            f"Angle Error: {h_error:.1f}° | Energy Error: {h_e_error:.1f} keV<br>"
            f"Points Earned: +{round_score} (Bonuses: {h_bonus})<br><br>"
            f"<b style='color:#ff3333;'>AI MODEL:</b><br>"
            f"Angle Error: {ai_error:.1f}° | Energy Error: {ai_e_error:.1f} keV<br>"
            f"Points Earned: +{ai_round_score}"
        )
        self.feedback_label.setText(feedback)
        self.update_hud()

    def update_hud(self):
        self.hud_label.setText(
            f"Track {self.current_track_idx + 1} / {len(self.round_tracks)}\n"
            f"Human Total: {self.total_score}\n"
            f"AI Total: {self.ai_total_score}"
        )

    def end_game(self):
        self.leaderboard.append({"name": self.player_name, "score": self.total_score, "rounds": len(self.round_tracks)})
        self.save_leaderboard()
        
        self.lb_table.setRowCount(len(self.leaderboard))
        for i, entry in enumerate(self.leaderboard):
            self.lb_table.setItem(i, 0, QTableWidgetItem(f"#{i+1}"))
            self.lb_table.setItem(i, 1, QTableWidgetItem(entry['name']))
            self.lb_table.setItem(i, 2, QTableWidgetItem(str(entry['score'])))
        
        self.stacked.setCurrentWidget(self.lb_screen)

    # ===============================
    # Plotting & Interactions
    # ===============================
    # ===============================
    # Plotting & Interactions
    # ===============================
    def on_pick(self, event):
        if self.locked_in: return 
        
        try:
            idx = event.ind[0] 
            row = self.round_tracks[self.current_track_idx]
            px, py, pz = row['raw_points'][idx]

            r = np.sqrt(px**2 + py**2 + pz**2)
            if r == 0: return

            elev = np.degrees(np.arcsin(pz / r))
            azim = np.degrees(np.arctan2(py, px))

            # Block signals so setting the slider values doesn't trigger two separate redraws
            self.elev_slider.blockSignals(True)
            self.azim_slider.blockSignals(True)
            self.elev_slider.setValue(int(elev))
            self.azim_slider.setValue(int(azim))
            self.elev_slider.blockSignals(False)
            self.azim_slider.blockSignals(False)
            
            self.update_plot_live()
            
        except Exception as e:
            print(f"Click missed: {e}")

    def init_plot(self):
        """Creates the plot and draws the vacancies ONCE per track."""
        row = self.round_tracks[self.current_track_idx]
        pts = row['raw_points']
        
        self.figure.clear()
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#1e1e1e')
        self.ax.xaxis.pane.fill = False; self.ax.yaxis.pane.fill = False; self.ax.zaxis.pane.fill = False

        self.ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='#00adb5', alpha=0.5, s=25, picker=5)
        self.ax.set_title(f"Vacancies: {len(pts)}", color='white', pad=20)

        # Fix the bounding box so zooming works properly
        if len(pts) > 0:
            max_r = np.max([pts[:,0].max()-pts[:,0].min(), pts[:,1].max()-pts[:,1].min(), pts[:,2].max()-pts[:,2].min()]) / 2.0
            mx, my, mz = np.mean(pts[:,0]), np.mean(pts[:,1]), np.mean(pts[:,2])
            self.ax.set_xlim(mx - max_r, mx + max_r)
            self.ax.set_ylim(my - max_r, my + max_r)
            self.ax.set_zlim(mz - max_r, mz + max_r)

        self.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        self.guess_quiver = None
        self.update_plot_live()

    def update_plot_live(self):
        """Erases the old yellow arrow and draws a new one without touching the camera view."""
        if self.locked_in: return
        if not hasattr(self, 'ax'): return

        row = self.round_tracks[self.current_track_idx]
        pts = row['raw_points']
        scale = np.max(np.abs(pts)) if len(pts) > 0 else 10

        elev = np.radians(self.elev_slider.value())
        azim = np.radians(self.azim_slider.value())
        gx, gy, gz = np.cos(elev)*np.cos(azim), np.cos(elev)*np.sin(azim), np.sin(elev)
        
        # Remove the previous arrow so we don't end up with thousands of them
        if self.guess_quiver is not None:
            self.guess_quiver.remove()
            
        self.guess_quiver = self.ax.quiver(0, 0, 0, gx*scale, gy*scale, gz*scale, color='yellow', linewidth=5, label='Your Vector Guess')
        
        # Fast redraw
        self.canvas.draw_idle()

    def draw_answers(self):
        """Draws the final AI and True vectors when locked in."""
        row = self.round_tracks[self.current_track_idx]
        pts = row['raw_points']
        scale = np.max(np.abs(pts)) if len(pts) > 0 else 10

        self.ax.quiver(0, 0, 0, row['true_vx']*scale, row['true_vy']*scale, row['true_vz']*scale, color='#00ff00', linewidth=3, label='True Vector')
        self.ax.quiver(0, 0, 0, row['pred_vx']*scale, row['pred_vy']*scale, row['pred_vz']*scale, color='#ff3333', linewidth=3, label='AI Vector')
        
        self.ax.set_title(f"True Energy: {row['true_energy']:.1f} keV", color='white', pad=20)
        self.ax.legend(facecolor='#2d2d2d', edgecolor='#3a3a3a', labelcolor='white')
        
        self.canvas.draw_idle()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VectorGame()
    window.show()
    sys.exit(app.exec_())