import sys
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QDoubleSpinBox, QSpinBox,
    QSlider, QCheckBox, QGroupBox, QFormLayout, QFrame
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Apply Matplotlib Dark Theme to match the UI
plt.style.use('dark_background')

# ===============================
# Load Data (Keep your existing paths)
# ===============================
RESULTS_CSV = "results/pipeline_b/pointnet_3d_eval_results.csv"
RAW_DATA_CSV = "data/mcpe-3d/mcpe_3d_eval_10k.csv" 

print("[INFO] Loading Results...")
try:
    df_results = pd.read_csv(RESULTS_CSV)
    print("[INFO] Loading Raw Data...")
    df_raw = pd.read_csv(RAW_DATA_CSV)
except FileNotFoundError:
    print("[ERROR] CSV files not found. Check your paths!")
    sys.exit()

# Extract raw points and vacancy counts
grouped = df_raw.groupby('ion_number')
raw_points_list = []
vacancy_counts = []
ion_numbers = []

for ion_idx, group in grouped:
    pts = group[['x', 'y', 'z']].values
    pts = pts - np.mean(pts, axis=0)  
    raw_points_list.append(pts)
    vacancy_counts.append(len(pts))
    ion_numbers.append(ion_idx)

df_results['ion_number'] = ion_numbers
df_results['num_vacancies'] = vacancy_counts
df_results['raw_points'] = raw_points_list

# Calculate Angular Error
true_vec = df_results[["true_vx","true_vy","true_vz"]].values
pred_vec = df_results[["pred_vx","pred_vy","pred_vz"]].values
true_vec = true_vec / (np.linalg.norm(true_vec, axis=1, keepdims=True) + 1e-9)
pred_vec = pred_vec / (np.linalg.norm(pred_vec, axis=1, keepdims=True) + 1e-9)
cos_theta = np.sum(true_vec * pred_vec, axis=1)
cos_theta = np.clip(cos_theta, -1.0, 1.0)
df_results["theta_deg"] = np.degrees(np.arccos(cos_theta))
df = df_results

# ===============================
# MODERN STYLESHEET (QSS)
# ===============================
MODERN_THEME = """
QWidget {
    background-color: #1e1e1e;
    color: #e0e0e0;
    font-family: 'Segoe UI', -apple-system, sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #3a3a3a;
    border-radius: 6px;
    margin-top: 15px;
    font-weight: bold;
    color: #00adb5;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
}
QPushButton {
    background-color: #2d2d2d;
    border: 1px solid #3a3a3a;
    border-radius: 5px;
    padding: 8px 12px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #3d3d3d;
    border: 1px solid #00adb5;
    color: #ffffff;
}
QPushButton:pressed {
    background-color: #00adb5;
    color: #000000;
}
QSpinBox, QDoubleSpinBox {
    background-color: #2d2d2d;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    padding: 4px;
    color: #ffffff;
}
QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #00adb5;
}
QSlider::groove:horizontal {
    border: 1px solid #3a3a3a;
    height: 6px;
    background: #2d2d2d;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #00adb5;
    width: 14px;
    margin: -4px 0;
    border-radius: 7px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid #3a3a3a;
    background: #2d2d2d;
}
QCheckBox::indicator:checked {
    background: #00adb5;
    border: 1px solid #00adb5;
}
"""

# ===============================
# Viewer UI
# ===============================
class Viewer(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MCPE 3D Pro Analysis Viewer")
        self.resize(1400, 900)
        self.setStyleSheet(MODERN_THEME)

        self.current_index = 0
        self.default_elev = 20
        self.default_azim = -60

        # MAIN LAYOUT: Horizontal (Sidebar + Plot)
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        self.setLayout(main_layout)

        # ---------------------------------------------------------
        # LEFT: SIDEBAR (Controls)
        # ---------------------------------------------------------
        sidebar = QWidget()
        sidebar.setFixedWidth(350)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(15)

        # Track Info Header
        self.track_label = QLabel("Loading...")
        self.track_label.setAlignment(Qt.AlignCenter)
        self.track_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffffff; padding: 10px; background-color: #2d2d2d; border-radius: 6px;")
        sidebar_layout.addWidget(self.track_label)

        # Navigation Buttons
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("◀ Previous")
        self.prev_button.clicked.connect(self.prev_track)
        self.next_button = QPushButton("Next ▶")
        self.next_button.clicked.connect(self.next_track)
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        sidebar_layout.addLayout(nav_layout)

        # Filters Group
        filter_group = QGroupBox("Data Filters")
        filter_form = QFormLayout()
        filter_form.setHorizontalSpacing(20)

        # --- ADDED ENERGY FILTERS HERE ---
        self.min_vac = QSpinBox(); self.min_vac.setRange(0, 10000); self.min_vac.setValue(0); self.min_vac.valueChanged.connect(self.reset_index)
        self.max_vac = QSpinBox(); self.max_vac.setRange(0, 10000); self.max_vac.setValue(10000); self.max_vac.valueChanged.connect(self.reset_index)
        
        self.min_energy = QDoubleSpinBox(); self.min_energy.setRange(0, 1000); self.min_energy.setValue(0); self.min_energy.valueChanged.connect(self.reset_index)
        self.max_energy = QDoubleSpinBox(); self.max_energy.setRange(0, 1000); self.max_energy.setValue(1000); self.max_energy.valueChanged.connect(self.reset_index)

        self.min_error = QDoubleSpinBox(); self.min_error.setRange(0, 180); self.min_error.setValue(0); self.min_error.valueChanged.connect(self.reset_index)
        self.max_error = QDoubleSpinBox(); self.max_error.setRange(0, 180); self.max_error.setValue(180); self.max_error.valueChanged.connect(self.reset_index)

        filter_form.addRow("Min Vacancies:", self.min_vac)
        filter_form.addRow("Max Vacancies:", self.max_vac)
        filter_form.addRow("Min Energy (keV):", self.min_energy)
        filter_form.addRow("Max Energy (keV):", self.max_energy)
        filter_form.addRow("Min Error (°):", self.min_error)
        filter_form.addRow("Max Error (°):", self.max_error)
        filter_group.setLayout(filter_form)
        sidebar_layout.addWidget(filter_group)

        # View Options Group
        view_group = QGroupBox("Render Options")
        view_layout = QVBoxLayout()
        
        self.analysis_mode_checkbox = QCheckBox("Analysis Mode (Points & Curve)")
        self.analysis_mode_checkbox.setChecked(True)
        self.analysis_mode_checkbox.stateChanged.connect(self.update_plot)
        view_layout.addWidget(self.analysis_mode_checkbox)

        self.lock_axes_checkbox = QCheckBox("Lock Cube Aspect Ratio")
        self.lock_axes_checkbox.setChecked(True)
        self.lock_axes_checkbox.stateChanged.connect(self.update_plot)
        view_layout.addWidget(self.lock_axes_checkbox)

        view_group.setLayout(view_layout)
        sidebar_layout.addWidget(view_group)

        # Camera Controls Group
        cam_group = QGroupBox("Camera Rotation")
        cam_layout = QVBoxLayout()
        
        cam_layout.addWidget(QLabel("Elevation"))
        self.elev_slider = QSlider(Qt.Horizontal); self.elev_slider.setRange(-90, 90); self.elev_slider.setValue(self.default_elev)
        self.elev_slider.valueChanged.connect(self.update_view)
        cam_layout.addWidget(self.elev_slider)

        cam_layout.addWidget(QLabel("Azimuth"))
        self.azim_slider = QSlider(Qt.Horizontal); self.azim_slider.setRange(-180, 180); self.azim_slider.setValue(self.default_azim)
        self.azim_slider.valueChanged.connect(self.update_view)
        cam_layout.addWidget(self.azim_slider)

        self.reset_view_button = QPushButton("Reset Camera Rotation")
        self.reset_view_button.clicked.connect(self.reset_view)
        cam_layout.addWidget(self.reset_view_button)

        cam_group.setLayout(cam_layout)
        sidebar_layout.addWidget(cam_group)

        # Push everything to the top
        sidebar_layout.addStretch()
        main_layout.addWidget(sidebar)

        # ---------------------------------------------------------
        # RIGHT: PLOT AREA
        # ---------------------------------------------------------
        plot_container = QWidget()
        plot_container.setStyleSheet("background-color: #1e1e1e; border-radius: 8px;")
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        # Create Matplotlib Figure (Seamless background color)
        self.figure = Figure(facecolor='#1e1e1e')
        self.canvas = FigureCanvas(self.figure)
        
        # ADD MATPLOTLIB TOOLBAR FOR ZOOMING
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("background-color: #2d2d2d; border: none; color: white;")
        
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        main_layout.addWidget(plot_container, stretch=1)

        # Initialize
        self.update_plot()

    # ===============================
    def get_filtered(self):
        # --- ADDED ENERGY LOGIC TO FILTER ---
        return df[
            (df.num_vacancies >= self.min_vac.value()) &
            (df.num_vacancies <= self.max_vac.value()) &
            (df.true_energy >= self.min_energy.value()) &
            (df.true_energy <= self.max_energy.value()) &
            (df.theta_deg >= self.min_error.value()) &
            (df.theta_deg <= self.max_error.value())
        ]

    def reset_index(self):
        self.current_index = 0
        self.update_plot()

    def next_track(self):
        self.current_index += 1
        self.update_plot()

    def prev_track(self):
        self.current_index -= 1
        self.update_plot()

    def reset_view(self):
        self.elev_slider.setValue(self.default_elev)
        self.azim_slider.setValue(self.default_azim)
        self.update_view()

    def update_view(self):
        if hasattr(self, "ax"):
            self.ax.view_init(elev=self.elev_slider.value(), azim=self.azim_slider.value())
            self.canvas.draw()

    # ===============================
    def update_plot(self):
        filtered = self.get_filtered()
        if len(filtered) == 0:
            self.track_label.setText("0 Tracks Found")
            self.figure.clear()
            self.canvas.draw()
            return

        self.current_index %= len(filtered)
        row = filtered.iloc[self.current_index]

        true_v = np.array([row.true_vx, row.true_vy, row.true_vz])
        pred_v = np.array([row.pred_vx, row.pred_vy, row.pred_vz])
        pts = row.raw_points

        self.figure.clear()
        self.ax = self.figure.add_subplot(111, projection='3d')
        
        # Style the 3D pane to look good in dark mode
        self.ax.set_facecolor('#1e1e1e')
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('#444444')
        self.ax.yaxis.pane.set_edgecolor('#444444')
        self.ax.zaxis.pane.set_edgecolor('#444444')

        is_analysis = self.analysis_mode_checkbox.isChecked()

        if is_analysis:
            # Vacancy Cloud (Cyan instead of dark blue to pop against dark background)
            self.ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='#00adb5', alpha=0.4, s=15, label="Vacancies")

            # Backbone Curve
            if len(pts) > 1:
                projections = np.dot(pts, true_v)
                sorted_indices = np.argsort(projections)
                sorted_pts = pts[sorted_indices]
                self.ax.plot(sorted_pts[:, 0], sorted_pts[:, 1], sorted_pts[:, 2], 
                             color='#ff00ff', alpha=0.6, linewidth=2, label="Track Core")

            scale = np.max(np.abs(pts)) if len(pts) > 0 else 10
        else:
            scale = 1.0 

        # Vectors
        self.ax.quiver(0, 0, 0, true_v[0]*scale, true_v[1]*scale, true_v[2]*scale,
                       color='#00ff00', linewidth=4, label="True Vector", arrow_length_ratio=0.1)
        self.ax.quiver(0, 0, 0, pred_v[0]*scale, pred_v[1]*scale, pred_v[2]*scale,
                       color='#ff3333', linewidth=4, label="Predicted Vector", arrow_length_ratio=0.1)

        # Formatting
        if self.lock_axes_checkbox.isChecked() and is_analysis and len(pts) > 0:
            max_range = np.array([pts[:,0].max()-pts[:,0].min(), pts[:,1].max()-pts[:,1].min(), pts[:,2].max()-pts[:,2].min()]).max() / 2.0
            mid_x, mid_y, mid_z = np.mean(pts[:,0]), np.mean(pts[:,1]), np.mean(pts[:,2])
            self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
            self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
            self.ax.set_zlim(mid_z - max_range, mid_z + max_range)

        self.ax.set_xlabel("X (Å)")
        self.ax.set_ylabel("Y (Å)")
        self.ax.set_zlabel("Z (Å)")
        
        # Modern legend
        legend = self.ax.legend(facecolor='#2d2d2d', edgecolor='#3a3a3a')
        for text in legend.get_texts(): text.set_color("white")

        # Update UI Labels
        self.figure.subplots_adjust(left=0, right=1, bottom=0, top=1) # Maximize plot area
        self.track_label.setText(
            f"Track {self.current_index+1} / {len(filtered)}\n"
            f"Ion: {row.ion_number} | {row.true_energy:.1f} keV\n"
            f"{row.num_vacancies} Vacs | Error: {row.theta_deg:.1f}°"
        )
        
        self.ax.view_init(elev=self.elev_slider.value(), azim=self.azim_slider.value())
        self.canvas.draw()

# ===============================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Viewer()
    window.show()
    sys.exit(app.exec_())