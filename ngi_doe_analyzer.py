"""
NGI DoE Analyzer v2
Design of Experiments — Manuel veri girişli analiz aracı
pyDOE3 + statsmodels + PyQt6
"""
import sys, os, json, math, datetime, itertools
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

import pyDOE3

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QCheckBox, QComboBox,
    QTabWidget, QScrollArea, QFrame, QFileDialog,
    QMessageBox, QDialog, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy, QSpacerItem, QGroupBox,
    QDialogButtonBox, QAbstractItemView, QSpinBox,
    QDoubleSpinBox, QRadioButton, QButtonGroup, QTextEdit
)
from PyQt6.QtCore import Qt, QSize, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon, QPalette

# ─── resource_path ────────────────────────────────────────────────────────────
def resource_path(rel):
    base = getattr(sys, '_MEIPASS',
        os.path.dirname(os.path.abspath(
            sys.executable if getattr(sys,'frozen',False) else __file__)))
    return os.path.join(base, rel)

# ─── Renkler & Stiller ────────────────────────────────────────────────────────
BG    = "#0e1219"
BG2   = "#141824"
BG3   = "#1c2336"
NAVY  = "#002D62"
GOLD  = "#FFC600"
TXT   = "#e0eaf8"
TXT2  = "#7090b0"
TEAL  = "#0a8a6a"
PURP  = "#6a20a0"
CP    = ["#2E75B6","#ED7D31","#70AD47","#E84040","#7030A0",
         "#00B0F0","#D4A000","#C00000","#00B050","#FF69B4"]

RESPONSE_LABELS = {
    "mmad":      "MMAD (µm)",
    "gsd":       "GSD",
    "fpd_5um":   "FPD <5µm (mg)",
    "fpf_5um":   "FPF <5µm (%)",
    "fpd_3um":   "FPD <3µm (mg)",
    "fpf_3um":   "FPF <3µm (%)",
    "fpd_15um":  "FPD <1.5µm (mg)",
    "fpf_15um":  "FPF <1.5µm (%)",
    "metered":   "Metered Doz (mg)",
    "delivered": "Delivered Doz (mg)",
}

DESIGN_TYPES = [
    "Full Factorial (2k)",
    "Fractional Factorial (2k-p)",
    "Central Composite (CCD/RSM)",
    "Box-Behnken (BBD)",
    "Plackett-Burman",
    "One Factor at a Time (OFAT)",
]

# Tasarım rehberi içerikleri
DESIGN_GUIDE = {
    "Full Factorial (2k)": {
        "ne_zaman": "Faktör sayısı ≤4 olduğunda ve tüm etkileşimleri görmek istediğinde.",
        "avantaj":  "Tüm ana etkiler ve etkileşimler hesaplanır. Sonuçlar kesin.",
        "dezavantaj": "Faktör sayısı arttıkça run sayısı katlanarak büyür (2⁵=32, 2⁶=64...).",
        "oneri":    "2-3 faktör için ideal başlangıç tasarımı.",
        "run_formula": lambda k, _: 2**k,
        "seviye": 2,
    },
    "Fractional Factorial (2k-p)": {
        "ne_zaman": "5+ faktörü hızlıca taramak istediğinde. Hangilerinin önemli olduğunu bilmiyorsun.",
        "avantaj":  "Full Factorial'a göre çok daha az run. Yüksek faktör sayısında kullanışlı.",
        "dezavantaj": "Bazı etkileşimler hesaplanamaz (aliasing). Tarama için uygundur, optimizasyon için değil.",
        "oneri":    "Önce bu tasarımla tara, anlamlı faktörleri bul, sonra CCD/BBD ile optimize et.",
        "run_formula": lambda k, _: max(8, 2**(k-1)),
        "seviye": 2,
    },
    "Central Composite (CCD/RSM)": {
        "ne_zaman": "2-4 faktörün optimumunu bulmak istediğinde. Eğrisel ilişki bekliyorsan.",
        "avantaj":  "5 seviye ile quadratic model kurulur. Gerçek optimumu yakalayabilir.",
        "dezavantaj": "α noktaları faktör sınırlarının dışına çıkabilir — formülasyon kısıtı varsa dikkat.",
        "oneri":    "Tarama sonrası anlamlı 2-3 faktörle kullan. En güçlü optimizasyon tasarımı.",
        "run_formula": lambda k, _: 2**k + 2*k + 4,
        "seviye": 5,
    },
    "Box-Behnken (BBD)": {
        "ne_zaman": "3-4 faktörün optimumunu bulmak istediğinde. Faktör sınırlarının dışına çıkmak istemiyorsan.",
        "avantaj":  "CCD'ye göre az run, köşe noktaları yok — aşırı kombinasyonları test etmez.",
        "dezavantaj": "En az 3 faktör gerektirir. 2 faktörle kullanılamaz.",
        "oneri":    "Formülasyon kısıtı olan çalışmalarda CCD'ye tercih edilir.",
        "run_formula": lambda k, _: 2*k*(k-1) + 3 if k>=3 else 0,
        "seviye": 3,
    },
    "Plackett-Burman": {
        "ne_zaman": "5+ faktörü minimum run ile taramak istediğinde.",
        "avantaj":  "Çok az run (N = 4'ün katı). 11 faktörü 12 run'da tarayabilirsin.",
        "dezavantaj": "Sadece ana etkiler hesaplanır, etkileşimler hesaplanamaz.",
        "oneri":    "İlk tarama çalışması için en verimli tasarım. Sonraki adımda CCD/BBD kullan.",
        "run_formula": lambda k, _: max(8, ((k+1)//4+1)*4),
        "seviye": 2,
    },
    "One Factor at a Time (OFAT)": {
        "ne_zaman": "Referans olarak veya tek bir faktörü izole etmek istediğinde.",
        "avantaj":  "Anlaşılması ve uygulanması kolay.",
        "dezavantaj": "Faktörler arası etkileşimleri göremez. DoE'nin temel avantajını kaybedersin.",
        "oneri":    "Karşılaştırma amaçlı kullan. Optimizasyon için tercih etme.",
        "run_formula": lambda k, _: 2*k + 1,
        "seviye": 2,
    },
}

STYLE = f"""
QMainWindow, QDialog {{
    background: {BG};
}}
QWidget {{
    background: {BG};
    color: {TXT};
    font-family: 'Segoe UI', 'Arial';
    font-size: 13px;
}}
QLabel {{ color: {TXT}; background: transparent; }}
QLineEdit, QDoubleSpinBox, QSpinBox, QTextEdit {{
    background: {BG3};
    border: 1px solid #2a4060;
    border-radius: 4px;
    padding: 3px 6px;
    color: {TXT};
}}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus {{ border: 1px solid {GOLD}; }}
QPushButton {{
    background: {BG3};
    border: 1px solid #2a4060;
    border-radius: 5px;
    padding: 5px 12px;
    color: {TXT};
    font-weight: 500;
}}
QPushButton:hover {{ background: #253a5e; border-color: {GOLD}; }}
QPushButton:pressed {{ background: {BG2}; }}
QTabWidget::pane {{
    border: 1px solid #2a4060;
    background: {BG2};
    border-radius: 4px;
}}
QTabBar::tab {{
    background: {BG3};
    color: {TXT2};
    border: 1px solid #2a4060;
    padding: 6px 16px;
    border-bottom: none;
    margin-right: 2px;
    border-radius: 4px 4px 0 0;
}}
QTabBar::tab:selected {{
    background: {BG2};
    color: {GOLD};
    border-color: {GOLD};
}}
QTabBar::tab:hover:!selected {{ background: #253a5e; color: {TXT}; }}
QComboBox {{
    background: {BG3};
    border: 1px solid #2a4060;
    border-radius: 4px;
    padding: 4px 8px;
    color: {TXT};
}}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background: {BG2};
    color: {TXT};
    selection-background-color: #1F4E79;
}}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{
    background: {BG2}; width: 8px; border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: #2a4060; border-radius: 4px; min-height: 20px;
}}
QTableWidget {{
    background: {BG2};
    gridline-color: #2a4060;
    color: {TXT};
    border: 1px solid #2a4060;
    border-radius: 4px;
}}
QTableWidget::item {{ padding: 4px; }}
QTableWidget::item:selected {{ background: #1F4E79; }}
QHeaderView::section {{
    background: #1F4E79;
    color: white;
    padding: 5px;
    border: 1px solid #2a4060;
    font-weight: bold;
}}
QGroupBox {{
    border: 1px solid #2a4060;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 6px;
    color: {TXT2};
    font-size: 11px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {GOLD};
}}
QCheckBox {{ color: {TXT}; background: transparent; spacing: 6px; }}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    border: 1px solid #2a4060;
    border-radius: 3px;
    background: {BG3};
}}
QCheckBox::indicator:checked {{
    background: #2E75B6;
    border-color: #4a95d6;
}}
QRadioButton {{ color: {TXT}; background: transparent; spacing: 6px; }}
QRadioButton::indicator {{
    width: 14px; height: 14px;
    border: 1px solid #2a4060;
    border-radius: 7px;
    background: {BG3};
}}
QRadioButton::indicator:checked {{
    background: {GOLD};
    border-color: {GOLD};
}}
"""

def make_btn(text, color=None, height=32):
    b = QPushButton(text)
    b.setFixedHeight(height)
    c = color or "rgba(30,50,90,0.8)"
    b.setStyleSheet(f"""
        QPushButton {{
            background: {c};
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 5px;
            padding: 4px 12px;
            color: {TXT};
            font-weight: 500;
        }}
        QPushButton:hover {{ background: rgba(255,255,255,0.08); border-color: {GOLD}; }}
        QPushButton:pressed {{ background: rgba(0,0,0,0.2); }}
    """)
    return b

def section_label(text):
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color: {GOLD}; font-size: 11px; font-weight: bold; background: transparent;")
    return lbl

def card_frame():
    f = QFrame()
    f.setStyleSheet(f"""
        QFrame {{
            background: {BG2};
            border: 1px solid #2a4060;
            border-radius: 8px;
        }}
    """)
    return f

# ═══════════════════════════════════════════════════════════════════════════════
# WORKER THREAD'LER — UI donmasını engeller
# ═══════════════════════════════════════════════════════════════════════════════
class OptimizeWorker(QThread):
    finished = pyqtSignal(object, float)  # (x_opt, d_opt)
    error    = pyqtSignal(str)

    def __init__(self, project, tgt_rows, resp_ranges):
        super().__init__()
        self.project     = project
        self.tgt_rows    = tgt_rows
        self.resp_ranges = resp_ranges

    def run(self):
        from scipy.optimize import differential_evolution
        all_factors = self.project.factors
        cont_idx    = [i for i,f in enumerate(all_factors) if f["type"]=="continuous"]
        bounds      = [(all_factors[i]["low"], all_factors[i]["high"]) for i in cont_idx]

        def desirability(x_cont):
            actual_vals = []
            cont_ptr = 0
            for i, f in enumerate(all_factors):
                if i in cont_idx:
                    actual_vals.append(x_cont[cont_ptr]); cont_ptr += 1
                else:
                    actual_vals.append(f.get("mid", (f["low"]+f["high"])/2))
            d_vals = []
            for resp, info in self.tgt_rows.items():
                if resp not in self.project.model_results: continue
                pred, _ = self.project.predict_at(resp, actual_vals)
                if pred is None: continue
                goal = info["combo"].currentText()
                lo, hi = self.resp_ranges.get(resp, (pred-1, pred+1))
                span = hi - lo if hi != lo else 1.0
                if goal == "Minimize Et":
                    d = max(0.0, min(1.0, (hi-pred)/span))
                elif goal == "Maximize Et":
                    d = max(0.0, min(1.0, (pred-lo)/span))
                else:
                    t = info["target"].value()
                    d = max(0.0, 1.0 - abs(pred-t)/span)
                d_vals.append(d)
            if not d_vals: return 1.0
            return -(np.prod(d_vals) ** (1.0/len(d_vals)))

        try:
            result = differential_evolution(
                desirability, bounds, seed=42,
                maxiter=800, tol=1e-7, polish=True,
                popsize=15, mutation=(0.5,1.5), recombination=0.7)
            self.finished.emit(result.x, -result.fun)
        except Exception as e:
            self.error.emit(str(e))


class SurfaceWorker(QThread):
    finished = pyqtSignal(object, object, object)  # XX, YY, ZZ
    error    = pyqtSignal(str)

    def __init__(self, project, resp_key, x_f, y_f, x_name, y_name, fixed_vals, n=40):
        super().__init__()
        self.project    = project
        self.resp_key   = resp_key
        self.x_f        = x_f
        self.y_f        = y_f
        self.x_name     = x_name
        self.y_name     = y_name
        self.fixed_vals = fixed_vals
        self.n          = n

    def run(self):
        try:
            n = self.n
            xs = np.linspace(self.x_f["low"], self.x_f["high"], n)
            ys = np.linspace(self.y_f["low"], self.y_f["high"], n)
            XX, YY = np.meshgrid(xs, ys)
            ZZ_flat = []
            for xi, yi in zip(XX.ravel(), YY.ravel()):
                actual_vals = []
                for f in self.project.factors:
                    if f["name"] == self.x_name:   actual_vals.append(xi)
                    elif f["name"] == self.y_name: actual_vals.append(yi)
                    elif f["name"] in self.fixed_vals:
                        actual_vals.append(self.fixed_vals[f["name"]])
                    else:
                        actual_vals.append(f.get("mid",(f["low"]+f["high"])/2))
                pred, _ = self.project.predict_at(self.resp_key, actual_vals)
                ZZ_flat.append(pred if pred is not None else float("nan"))
            ZZ = np.array(ZZ_flat).reshape(n, n)
            self.finished.emit(XX, YY, ZZ)
        except Exception as e:
            self.error.emit(str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# VERİ MODELİ
# ═══════════════════════════════════════════════════════════════════════════════
class DoEProject:
    def __init__(self):
        self.reset()

    def reset(self):
        self.factors       = []
        self.responses     = []
        self.design_type   = DESIGN_TYPES[0]
        self.design_matrix = None
        self.run_results   = {}
        self.model_results = {}
        self.model_errors  = {}
        self.product       = ""
        self.batch         = ""
        self.analyst       = ""
        self.date          = ""
        self.notes         = ""

    def get_factor_names(self):
        return [f["name"] for f in self.factors]

    def get_safe_names(self):
        return [f"X{i+1}" for i in range(len(self.factors))]

    def get_coded_df(self):
        df = self.design_matrix
        if df is None: return None, []
        safe_names = self.get_safe_names()
        coded_cols = {}
        for i, f in enumerate(self.factors):
            sn = safe_names[i]
            coded_col = f["name"] + "_coded"
            if coded_col in df.columns:
                coded_cols[sn] = df[coded_col].values
            elif f["type"] == "continuous" and f["high"] != f["low"]:
                actual = df[f["name"]].values
                lo, hi = f["low"], f["high"]
                mid = f.get("mid", (lo + hi) / 2)
                # 3 noktalı (-1, 0, +1) veya 2 noktalı (-1, +1) dönüşüm
                coded = np.where(
                    actual <= mid,
                    (actual - lo) / (mid - lo) - 1,
                    (actual - mid) / (hi - mid)
                )
                coded_cols[sn] = coded
            else:
                coded_cols[sn] = df[f["name"]].values
        return pd.DataFrame(coded_cols), safe_names

    def build_response_table(self):
        if self.design_matrix is None: return None
        df = self.design_matrix.copy()
        for resp in self.responses:
            df[resp] = [self.run_results.get(i, {}).get(resp, np.nan)
                        for i in range(len(df))]
        return df

    def fit_models(self):
        base_df, safe_names = self.get_coded_df()
        if base_df is None: return
        self.model_results = {}
        self.model_errors  = {}

        # Tasarım tipine göre hangi terimler modele girer
        use_interactions = self.design_type not in [
            "Plackett-Burman", "One Factor at a Time (OFAT)"]
        use_quadratic = self.design_type in [
            "Central Composite (CCD/RSM)", "Box-Behnken (BBD)"]

        for resp in self.responses:
            y_vals = [self.run_results.get(i, {}).get(resp, np.nan)
                      for i in range(len(base_df))]
            sub = base_df.copy()
            sub[resp] = y_vals
            sub = sub.dropna()

            terms = safe_names[:]
            if use_interactions:
                for a, b in itertools.combinations(safe_names, 2):
                    col = f"{a}_x_{b}"
                    sub[col] = sub[a] * sub[b]
                    terms.append(col)
            if use_quadratic:
                for a in safe_names:
                    col = f"{a}_sq"
                    sub[col] = sub[a] ** 2
                    terms.append(col)

            n_needed = len(terms) + 2
            if len(sub) < n_needed:
                self.model_errors[resp] = (
                    f"Yetersiz veri: {len(sub)} run var, "
                    f"en az {n_needed} gerekli "
                    f"({len(terms)} terim + 2).")
                continue
            try:
                formula = f"{resp} ~ " + " + ".join(terms)
                model = ols(formula, data=sub).fit()
                self.model_results[resp] = model
            except Exception as e:
                self.model_errors[resp] = str(e)

    def predict_at(self, resp, factor_actual_values):
        model = self.model_results.get(resp)
        if not model: return None, None
        safe_names = self.get_safe_names()
        row = {}
        for i, f in enumerate(self.factors):
            sn = safe_names[i]
            actual = factor_actual_values[i]
            lo, hi = f["low"], f["high"]
            mid = f.get("mid", (lo + hi) / 2)
            if f["type"] == "continuous" and hi != lo:
                if actual <= mid and (mid - lo) != 0:
                    coded = (actual - lo) / (mid - lo) - 1
                elif (hi - mid) != 0:
                    coded = (actual - mid) / (hi - mid)
                else:
                    coded = 0.0
            else:
                coded = actual
            row[sn] = coded
        for a, b in itertools.combinations(safe_names, 2):
            row[f"{a}_x_{b}"] = row[a] * row[b]
        if self.design_type in ["Central Composite (CCD/RSM)", "Box-Behnken (BBD)"]:
            for a in safe_names:
                row[f"{a}_sq"] = row[a] ** 2
        pred_df = pd.DataFrame([row])
        try:
            pred = float(model.predict(pred_df).iloc[0])
            try:
                se = float(np.sqrt(model.mse_resid)) if model.mse_resid else 0
                lo_ci, hi_ci = pred - 1.96*se, pred + 1.96*se
            except:
                lo_ci, hi_ci = pred, pred
            return pred, (lo_ci, hi_ci)
        except Exception as e:
            print(f"Predict error ({resp}): {e}")
            return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# FAKTÖR SATIRI WIDGET
# ═══════════════════════════════════════════════════════════════════════════════
class FactorRow(QWidget):
    def __init__(self, idx, parent=None):
        super().__init__(parent)
        self.idx = idx
        self.show_mid = False
        self.setStyleSheet("background: transparent;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(6)

        # Index
        lbl = QLabel(f"{idx+1}.")
        lbl.setFixedWidth(20)
        lbl.setStyleSheet(f"color:{TXT2}; font-size:11px; background:transparent;")
        lay.addWidget(lbl)

        # Ad
        self.e_name = QLineEdit(f"Faktör {idx+1}")
        self.e_name.setFixedWidth(130)
        lay.addWidget(self.e_name)

        # Tip
        self.combo_type = QComboBox()
        self.combo_type.addItems(["Sürekli", "Kategorik"])
        self.combo_type.setFixedWidth(90)
        self.combo_type.currentTextChanged.connect(self._on_type_change)
        lay.addWidget(self.combo_type)

        # Alt
        self.e_low = QLineEdit("0")
        self.e_low.setFixedWidth(65)
        self.e_low.setPlaceholderText("Alt")
        lay.addWidget(self.e_low)

        # Merkez (CCD/BBD için)
        self.lbl_mid = QLabel("Ort:")
        self.lbl_mid.setStyleSheet(f"color:{TXT2}; font-size:11px; background:transparent;")
        self.lbl_mid.setFixedWidth(25)
        self.e_mid = QLineEdit("")
        self.e_mid.setFixedWidth(65)
        self.e_mid.setPlaceholderText("oto")
        self.e_mid.setToolTip("Boş bırakılırsa (Alt+Üst)/2 kullanılır")
        self.lbl_mid.setVisible(False)
        self.e_mid.setVisible(False)
        lay.addWidget(self.lbl_mid)
        lay.addWidget(self.e_mid)

        # Üst
        self.e_high = QLineEdit("1")
        self.e_high.setFixedWidth(65)
        self.e_high.setPlaceholderText("Üst")
        lay.addWidget(self.e_high)

        # Birim
        self.e_unit = QLineEdit("")
        self.e_unit.setFixedWidth(55)
        self.e_unit.setPlaceholderText("Birim")
        lay.addWidget(self.e_unit)

        # Sil
        self.btn_del = QPushButton("✕")
        self.btn_del.setFixedSize(26, 26)
        self.btn_del.setStyleSheet("""
            QPushButton { background: rgba(100,20,20,0.6); border: 1px solid #6a2020;
                          border-radius: 4px; color: #ff8080; font-weight: bold; }
            QPushButton:hover { background: rgba(160,30,30,0.8); }
        """)
        lay.addWidget(self.btn_del)
        lay.addStretch()

    def set_show_mid(self, show):
        self.show_mid = show
        is_cont = self.combo_type.currentText() == "Sürekli"
        self.lbl_mid.setVisible(show and is_cont)
        self.e_mid.setVisible(show and is_cont)

    def _on_type_change(self, t):
        is_cont = (t == "Sürekli")
        self.e_low.setVisible(is_cont)
        self.e_high.setVisible(is_cont)
        self.lbl_mid.setVisible(self.show_mid and is_cont)
        self.e_mid.setVisible(self.show_mid and is_cont)

    def get_data(self):
        is_cont = self.combo_type.currentText() == "Sürekli"
        try: low = float(self.e_low.text().replace(",", "."))
        except: low = 0.0
        try: high = float(self.e_high.text().replace(",", "."))
        except: high = 1.0
        mid_txt = self.e_mid.text().strip()
        try: mid = float(mid_txt.replace(",", ".")) if mid_txt else (low + high) / 2
        except: mid = (low + high) / 2
        return {
            "name":  self.e_name.text().strip() or f"F{self.idx+1}",
            "type":  "continuous" if is_cont else "categorical",
            "low":   low,
            "mid":   mid,
            "high":  high,
            "unit":  self.e_unit.text().strip(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TASARIM REHBERİ WIDGET
# ═══════════════════════════════════════════════════════════════════════════════
class DesignGuideWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background: {BG3};
                border: 1px solid #1F4E79;
                border-radius: 8px;
            }}
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(6)

        hdr = QHBoxLayout()
        ico = QLabel("📖")
        ico.setStyleSheet("font-size:14px; background:transparent;")
        hdr.addWidget(ico)
        self.lbl_title = QLabel("Tasarım Rehberi")
        self.lbl_title.setStyleSheet(
            f"color:{GOLD}; font-size:12px; font-weight:bold; background:transparent;")
        hdr.addWidget(self.lbl_title)
        hdr.addStretch()
        self.lbl_run = QLabel("")
        self.lbl_run.setStyleSheet(
            f"color:#70e870; font-size:11px; font-weight:bold; background:transparent;")
        hdr.addWidget(self.lbl_run)
        lay.addLayout(hdr)

        self.lbl_ne_zaman = QLabel("")
        self.lbl_avantaj  = QLabel("")
        self.lbl_dezavantaj = QLabel("")
        self.lbl_oneri    = QLabel("")

        for lbl in [self.lbl_ne_zaman, self.lbl_avantaj,
                    self.lbl_dezavantaj, self.lbl_oneri]:
            lbl.setWordWrap(True)
            lbl.setStyleSheet(f"color:{TXT}; font-size:11px; background:transparent;")
            lay.addWidget(lbl)

    def update(self, design_type, k):
        info = DESIGN_GUIDE.get(design_type)
        if not info: return
        try:
            n_run = info["run_formula"](k, None)
        except:
            n_run = "?"
        seviye = info["seviye"]
        self.lbl_title.setText(f"📖  {design_type}")
        self.lbl_run.setText(
            f"~{n_run} run  |  {seviye} seviye" if k > 0 else "Faktör giriniz")
        self.lbl_ne_zaman.setText(f"🕐  Ne zaman:  {info['ne_zaman']}")
        self.lbl_avantaj.setText(f"✔  Avantaj:  {info['avantaj']}")
        self.lbl_dezavantaj.setText(f"⚠  Dezavantaj:  {info['dezavantaj']}")
        self.lbl_oneri.setText(f"💡  Öneri:  {info['oneri']}")


# ═══════════════════════════════════════════════════════════════════════════════
# SEKME 1 — PROJE & FAKTÖRLER
# ═══════════════════════════════════════════════════════════════════════════════
class Tab1_Factors(QWidget):
    def __init__(self, project: DoEProject, app_ref, parent=None):
        super().__init__(parent)
        self.project = project
        self.app = app_ref
        self.factor_rows = []
        self._build()

    def _build(self):
        main = QVBoxLayout(self)
        main.setSpacing(10)
        main.setContentsMargins(14, 14, 14, 14)

        # ── Proje bilgisi ─────────────────────────────────────────────────────
        info_card = card_frame()
        fl = QFormLayout(info_card)
        fl.setContentsMargins(12, 10, 12, 10)
        fl.setSpacing(6)
        fl.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        lbl_s = f"color:{TXT2}; font-size:11px; background:transparent;"
        def qlbl(t): l = QLabel(t); l.setStyleSheet(lbl_s); return l
        self.e_product = QLineEdit(); self.e_product.setPlaceholderText("Ürün adı")
        self.e_batch   = QLineEdit(); self.e_batch.setPlaceholderText("Lot No.")
        self.e_analyst = QLineEdit(); self.e_analyst.setPlaceholderText("Analist")
        self.e_date    = QLineEdit(datetime.date.today().strftime("%d.%m.%Y"))
        fl.addRow(qlbl("Ürün:"),    self.e_product)
        fl.addRow(qlbl("Lot:"),     self.e_batch)
        fl.addRow(qlbl("Analist:"), self.e_analyst)
        fl.addRow(qlbl("Tarih:"),   self.e_date)
        main.addWidget(info_card)

        # ── Tasarım tipi + rehber ─────────────────────────────────────────────
        design_row = QHBoxLayout(); design_row.setSpacing(10)
        design_card = card_frame()
        dl = QVBoxLayout(design_card); dl.setContentsMargins(12, 10, 12, 10); dl.setSpacing(6)
        top_dl = QHBoxLayout()
        top_dl.addWidget(section_label("⚗️  Tasarım Tipi:"))
        self.combo_design = QComboBox()
        self.combo_design.addItems(DESIGN_TYPES)
        self.combo_design.setFixedWidth(280)
        top_dl.addWidget(self.combo_design)
        top_dl.addStretch()
        dl.addLayout(top_dl)
        design_row.addWidget(design_card, 1)

        self.guide_widget = DesignGuideWidget()
        design_row.addWidget(self.guide_widget, 2)
        main.addLayout(design_row)

        self.combo_design.currentTextChanged.connect(self._on_design_change)

        # ── Faktörler ─────────────────────────────────────────────────────────
        factor_card = card_frame()
        fcl = QVBoxLayout(factor_card)
        fcl.setContentsMargins(12, 10, 12, 10)
        fcl.setSpacing(6)

        hdr = QHBoxLayout()
        hdr.addWidget(section_label("🔬  Faktörler (Bağımsız Değişkenler)"))
        hdr.addStretch()
        btn_add = make_btn("＋ Faktör Ekle", "rgba(20,50,100,0.7)", 28)
        btn_add.clicked.connect(self._add_factor_row)
        hdr.addWidget(btn_add)
        fcl.addLayout(hdr)

        # Sütun başlıkları
        hdr_row = QWidget(); hdr_row.setStyleSheet("background:transparent;")
        hr = QHBoxLayout(hdr_row); hr.setContentsMargins(0, 0, 0, 0); hr.setSpacing(6)
        self.col_headers = []
        for txt, w in [("#","20"),("Faktör Adı","130"),("Tip","90"),
                       ("Alt","65"),("","25"),("Merkez","65"),("Üst","65"),("Birim","55")]:
            l = QLabel(txt)
            l.setStyleSheet(
                f"color:{GOLD}; font-size:10px; font-weight:bold; background:transparent;")
            l.setFixedWidth(int(w))
            hr.addWidget(l)
            self.col_headers.append(l)
        hr.addStretch()
        fcl.addWidget(hdr_row)

        self.factor_scroll = QScrollArea()
        self.factor_scroll.setWidgetResizable(True)
        self.factor_scroll.setFixedHeight(190)
        self.factor_scroll.setStyleSheet("background:transparent; border:none;")
        self.factor_container = QWidget()
        self.factor_container.setStyleSheet("background:transparent;")
        self.factor_layout = QVBoxLayout(self.factor_container)
        self.factor_layout.setSpacing(2)
        self.factor_layout.setContentsMargins(0, 0, 0, 0)
        self.factor_layout.addStretch()
        self.factor_scroll.setWidget(self.factor_container)
        fcl.addWidget(self.factor_scroll)
        main.addWidget(factor_card)

        # ── Yanıt değişkenleri ────────────────────────────────────────────────
        resp_card = card_frame()
        rcl = QVBoxLayout(resp_card)
        rcl.setContentsMargins(12, 10, 12, 10)
        rcl.setSpacing(8)
        rcl.addWidget(section_label("🎯  Yanıt Değişkenleri (NGI Parametreleri)"))
        self.resp_checks = {}
        resp_grid = QGridLayout(); resp_grid.setSpacing(6)
        for i, (key, lbl) in enumerate(RESPONSE_LABELS.items()):
            cb = QCheckBox(lbl)
            cb.setStyleSheet("background: transparent;")
            self.resp_checks[key] = cb
            resp_grid.addWidget(cb, i // 4, i % 4)
        rcl.addLayout(resp_grid)
        main.addWidget(resp_card)

        # ── Alt buton ─────────────────────────────────────────────────────────
        btn_bar = QHBoxLayout(); btn_bar.addStretch()
        self.btn_build = make_btn("⚗️  Tasarım Matrisini Oluştur",
                                  "rgba(20,80,20,0.8)", 36)
        self.btn_build.clicked.connect(self._build_design)
        btn_bar.addWidget(self.btn_build)
        main.addLayout(btn_bar)
        main.addStretch()

        # Başlangıç
        self._add_factor_row()
        self._add_factor_row()
        self._on_design_change(self.combo_design.currentText())

    def _on_design_change(self, design):
        needs_mid = design in ["Central Composite (CCD/RSM)", "Box-Behnken (BBD)"]
        for row in self.factor_rows:
            row.set_show_mid(needs_mid)
        # Merkez başlığını göster/gizle
        self.col_headers[4].setVisible(needs_mid)  # "Ort:" label
        self.col_headers[5].setVisible(needs_mid)  # "Merkez"
        # Rehberi güncelle
        self.guide_widget.update(design, len(self.factor_rows))

    def _add_factor_row(self):
        design = self.combo_design.currentText()
        needs_mid = design in ["Central Composite (CCD/RSM)", "Box-Behnken (BBD)"]
        row = FactorRow(len(self.factor_rows))
        row.set_show_mid(needs_mid)
        row.btn_del.clicked.connect(lambda _, r=row: self._del_factor_row(r))
        self.factor_rows.append(row)
        self.factor_layout.insertWidget(self.factor_layout.count() - 1, row)
        self.guide_widget.update(self.combo_design.currentText(), len(self.factor_rows))

    def _del_factor_row(self, row):
        if len(self.factor_rows) <= 1:
            QMessageBox.warning(self, "", "En az 1 faktör olmalı."); return
        self.factor_rows.remove(row)
        row.setParent(None)
        for i, r in enumerate(self.factor_rows): r.idx = i
        self.guide_widget.update(self.combo_design.currentText(), len(self.factor_rows))

    def _build_design(self):
        factors = [r.get_data() for r in self.factor_rows]
        if not factors:
            QMessageBox.warning(self, "", "Faktör ekleyin."); return
        responses = [k for k, cb in self.resp_checks.items() if cb.isChecked()]
        if not responses:
            QMessageBox.warning(self, "", "En az 1 yanıt değişkeni seçin."); return
        design = self.combo_design.currentText()
        k = len(factors)
        try:
            matrix = self._generate_matrix(design, factors)
        except Exception as e:
            QMessageBox.critical(self, "Tasarım Hatası", str(e)); return

        # DataFrame
        cols = [f["name"] for f in factors]
        df = pd.DataFrame(matrix, columns=cols)

        # Coded → actual dönüşümü
        for f in factors:
            if f["type"] == "continuous" and f["name"] in df.columns:
                col = df[f["name"]]
                lo, mid, hi = f["low"], f["mid"], f["high"]
                # coded: -1 → low, 0 → mid, +1 → high
                actual = np.where(
                    col <= 0,
                    lo + (col + 1) * (mid - lo),
                    mid + col * (hi - mid)
                )
                df[f["name"] + "_coded"] = col
                df[f["name"]] = np.round(actual, 4)

        df.insert(0, "Run", range(1, len(df) + 1))

        # Projeye kaydet — eski veriyi temizle
        self.project.factors       = factors
        self.project.responses     = responses
        self.project.design_type   = design
        self.project.design_matrix = df
        self.project.run_results   = {}
        self.project.model_results = {}
        self.project.model_errors  = {}
        self.project.product  = self.e_product.text()
        self.project.batch    = self.e_batch.text()
        self.project.analyst  = self.e_analyst.text()
        self.project.date     = self.e_date.text()

        self.app.refresh_all_tabs()
        self.app.tabs.setCurrentIndex(1)
        QMessageBox.information(self, "✔ Tasarım Oluşturuldu",
            f"{design}\n{len(df)} run  |  {k} faktör  |  {len(responses)} yanıt\n\n"
            "Tasarım Matrisi sekmesine geçildi.")

    def _generate_matrix(self, design, factors):
        k = len(factors)
        cont = [f for f in factors if f["type"] == "continuous"]
        kc = len(cont)

        if design == "Full Factorial (2k)":
            return pyDOE3.ff2n(k)

        elif design == "Fractional Factorial (2k-p)":
            if k <= 2: return pyDOE3.ff2n(k)
            gens = {3:"a b ab", 4:"a b c abc", 5:"a b c d abcd"}
            gen = gens.get(k)
            return pyDOE3.fracfact(gen) if gen else pyDOE3.ff2n(k)

        elif design == "Central Composite (CCD/RSM)":
            if kc < 2:
                raise ValueError("CCD için en az 2 sürekli faktör gerekli.")
            return pyDOE3.ccdesign(kc, center=(4, 4), face="circumscribed")

        elif design == "Box-Behnken (BBD)":
            if kc < 3:
                raise ValueError("Box-Behnken için en az 3 sürekli faktör gerekli.")
            return pyDOE3.bbdesign(kc, center=3)

        elif design == "Plackett-Burman":
            n_runs = max(8, ((k + 1) // 4 + 1) * 4)
            return pyDOE3.pbdesign(n_runs)[:, :k]

        elif design == "One Factor at a Time (OFAT)":
            center = np.zeros((1, k))
            runs = [center]
            for i in range(k):
                lo = np.zeros((1, k)); lo[0, i] = -1
                hi = np.zeros((1, k)); hi[0, i] =  1
                runs += [lo, hi]
            return np.vstack(runs)
        else:
            return pyDOE3.ff2n(k)


# ═══════════════════════════════════════════════════════════════════════════════
# SEKME 2 — TASARIM MATRİSİ & VERİ GİRİŞİ
# ═══════════════════════════════════════════════════════════════════════════════
class Tab2_Design(QWidget):
    def __init__(self, project: DoEProject, app_ref, parent=None):
        super().__init__(parent)
        self.project = project
        self.app = app_ref
        self._col_blocking = False
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setSpacing(10)
        lay.setContentsMargins(14, 14, 14, 14)

        bar = QHBoxLayout(); bar.setSpacing(8)
        bar.addWidget(section_label("📋  Tasarım Matrisi & Yanıt Değerleri"))
        bar.addStretch()
        self.btn_randomize = make_btn("🔀 Randomize", "rgba(60,40,10,0.8)", 28)
        self.btn_randomize.clicked.connect(self._randomize)
        bar.addWidget(self.btn_randomize)
        self.btn_export = make_btn("⬇ Excel'e Aktar", "rgba(10,60,20,0.8)", 28)
        self.btn_export.clicked.connect(self._export_excel)
        bar.addWidget(self.btn_export)
        lay.addLayout(bar)

        self.lbl_info = QLabel(
            "Tasarım matrisi henüz oluşturulmadı. Faktörler sekmesinden başlayın.")
        self.lbl_info.setStyleSheet(
            f"color:{TXT2}; font-size:11px; background:transparent;")
        lay.addWidget(self.lbl_info)

        # Veri girişi ipucu
        tip = QLabel(
            "💡  Sarı sütunlara NGI ölçüm sonuçlarını girin. "
            "Hücreye tıklayıp değeri yazın, Enter ile sonraki satıra geçin.")
        tip.setStyleSheet(
            f"color:{TXT2}; font-size:10px; font-style:italic; background:transparent;")
        tip.setWordWrap(True)
        lay.addWidget(tip)

        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(
            self.table.styleSheet() +
            "QTableWidget { alternate-background-color: #181f2e; }")
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self.table.cellChanged.connect(self._on_cell_changed)
        lay.addWidget(self.table, 1)

    def refresh(self):
        try:
            self._col_blocking = True
            df = self.project.design_matrix
            if df is None:
                self.lbl_info.setText(
                    "Tasarım matrisi henüz oluşturulmadı.")
                self.table.setRowCount(0)
                self.table.setColumnCount(0)
                self._col_blocking = False
                return

            show_cols = [c for c in df.columns if not c.endswith("_coded")]
            all_cols  = show_cols + self.project.responses
            n_runs    = len(df)

            self.lbl_info.setText(
                f"Tasarım: {self.project.design_type}  |  "
                f"{n_runs} run  |  {len(self.project.factors)} faktör  |  "
                f"{len(self.project.responses)} yanıt  —  "
                f"Sarı sütunlara ölçüm değerlerini girin.")

            self.table.setColumnCount(len(all_cols))
            self.table.setRowCount(n_runs)
            self.table.setHorizontalHeaderLabels(
                [RESPONSE_LABELS.get(c, c) for c in all_cols])

            for ri in range(n_runs):
                for ci, col in enumerate(all_cols):
                    if col in df.columns:
                        val = df.iloc[ri][col]
                        if col == "Run":
                            txt = str(int(val))
                        elif isinstance(val, float):
                            txt = f"{val:.4f}"
                        else:
                            txt = str(val)
                        item = QTableWidgetItem(txt)
                        item.setFlags(
                            item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                        item.setForeground(QColor(TXT2))
                    else:
                        val = self.project.run_results.get(ri, {}).get(col, "")
                        txt = (f"{val:.4f}"
                               if isinstance(val, float) and not math.isnan(val)
                               else "")
                        item = QTableWidgetItem(txt)
                        item.setBackground(QColor("#1a2a1a"))
                        item.setForeground(QColor(GOLD))
                    self.table.setItem(ri, ci, item)

            # Yanıt sütun başlıklarını altın renge boya
            for ci in range(len(show_cols), len(all_cols)):
                h = self.table.horizontalHeaderItem(ci)
                if h: h.setForeground(QColor(GOLD))

            self._col_blocking = False
        except Exception as e:
            self._col_blocking = False
            print(f"Tab2 refresh error: {e}")

    def _on_cell_changed(self, row, col):
        if self._col_blocking: return
        df = self.project.design_matrix
        if df is None: return
        show_cols = [c for c in df.columns if not c.endswith("_coded")]
        resp_start = len(show_cols)
        if col < resp_start: return
        resp_idx = col - resp_start
        if resp_idx >= len(self.project.responses): return
        resp_key = self.project.responses[resp_idx]
        item = self.table.item(row, col)
        if not item: return
        try:
            val = float(item.text().replace(",", "."))
            if row not in self.project.run_results:
                self.project.run_results[row] = {}
            self.project.run_results[row][resp_key] = val
        except:
            pass

    def _randomize(self):
        df = self.project.design_matrix
        if df is None: return
        perm = np.random.permutation(len(df))
        self.project.design_matrix = (
            df.iloc[perm].reset_index(drop=True))
        self.project.design_matrix["Run"] = range(1, len(df) + 1)
        self.project.run_results = {}
        self.refresh()

    def _export_excel(self):
        df = self.project.design_matrix
        if df is None: return
        path, _ = QFileDialog.getSaveFileName(
            self, "Excel Kaydet",
            f"DoE_Matrix_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx",
            "Excel (*.xlsx)")
        if not path: return
        try:
            export_df = self.project.build_response_table() or df
            export_df.to_excel(path, index=False)
            QMessageBox.information(self, "✔", "Excel kaydedildi.")
        except Exception as e:
            QMessageBox.critical(self, "Hata", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# SEKME 3 — MODEL & ANOVA
# ═══════════════════════════════════════════════════════════════════════════════
class Tab3_Analysis(QWidget):
    def __init__(self, project: DoEProject, app_ref, parent=None):
        super().__init__(parent)
        self.project = project
        self.app = app_ref
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setSpacing(10)
        lay.setContentsMargins(14, 14, 14, 14)

        bar = QHBoxLayout(); bar.setSpacing(8)
        bar.addWidget(section_label("📊  Model & ANOVA"))
        bar.addStretch()
        bar.addWidget(QLabel("Yanıt:"))
        self.combo_resp = QComboBox(); self.combo_resp.setFixedWidth(220)
        self.combo_resp.currentTextChanged.connect(self._show_selected)
        bar.addWidget(self.combo_resp)
        self.btn_fit = make_btn("▶ Model Fit", "rgba(20,80,20,0.8)", 32)
        self.btn_fit.clicked.connect(self._fit)
        bar.addWidget(self.btn_fit)
        lay.addLayout(bar)

        spl = QSplitter(Qt.Orientation.Horizontal)

        # Sol: ANOVA + özet
        left = QWidget()
        ll = QVBoxLayout(left); ll.setContentsMargins(0, 0, 6, 0); ll.setSpacing(8)
        ll.addWidget(section_label("ANOVA Tablosu"))
        self.anova_table = QTableWidget()
        self.anova_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        ll.addWidget(self.anova_table)
        ll.addWidget(section_label("Model Özeti"))
        self.txt_summary = QTextEdit()
        self.txt_summary.setReadOnly(True)
        self.txt_summary.setFixedHeight(140)
        self.txt_summary.setStyleSheet(f"""
            QTextEdit {{ background:{BG3}; border:1px solid #2a4060;
                         border-radius:4px; color:{TXT};
                         font-family:Consolas,monospace; font-size:11px; }}
        """)
        ll.addWidget(self.txt_summary)
        spl.addWidget(left)

        # Sağ: grafikler
        right = QWidget()
        rl = QVBoxLayout(right); rl.setContentsMargins(6, 0, 0, 0)
        self.fig_analysis = Figure(facecolor=BG2, tight_layout=True)
        self.canvas_analysis = FigureCanvas(self.fig_analysis)
        rl.addWidget(self.canvas_analysis)
        spl.addWidget(right)
        spl.setSizes([420, 560])
        lay.addWidget(spl, 1)

    def refresh(self):
        try:
            self.fig_analysis.clear()
            self.canvas_analysis.draw()
            self.anova_table.setRowCount(0)
            self.txt_summary.clear()
        except Exception as e:
            print(f"Tab3 figure clear error: {e}")
        self.combo_resp.clear()
        for r in self.project.responses:
            self.combo_resp.addItem(RESPONSE_LABELS.get(r, r), r)

    def _fit(self):
        if self.project.design_matrix is None:
            QMessageBox.warning(
                self, "", "Önce tasarım matrisi oluşturun."); return
        has_data = any(
            self.project.run_results.get(i, {})
            for i in range(len(self.project.design_matrix)))
        if not has_data:
            QMessageBox.warning(
                self, "",
                "Tasarım Matrisi sekmesinden yanıt değerlerini girin."); return
        self.project.fit_models()
        errors = self.project.model_errors
        if errors:
            msg = "\n".join([f"  {k}: {v}" for k, v in errors.items()])
            QMessageBox.warning(self, "Model Uyarıları",
                f"Bazı modeller fit edilemedi:\n{msg}")
        if self.project.model_results:
            self._show_selected()
            QMessageBox.information(self, "✔ Model Fit",
                f"{len(self.project.model_results)} model başarıyla fit edildi.")

    def _show_selected(self):
        key = self.combo_resp.currentData()
        if not key: return
        model = self.project.model_results.get(key)
        err   = self.project.model_errors.get(key)
        if not model:
            msg = err if err else "Model henüz fit edilmedi. '▶ Model Fit' butonuna basın."
            self.txt_summary.setText(msg)
            self.anova_table.setRowCount(0)
            self.fig_analysis.clear()
            self.canvas_analysis.draw()
            return
        try:
            anova = anova_lm(model, typ=2)
            self._fill_anova(anova)
        except Exception as e:
            print(f"ANOVA error: {e}")
        try:
            r2 = model.rsquared; r2a = model.rsquared_adj
            rmse = float(np.sqrt(model.mse_resid)) if model.mse_resid else 0
            self.txt_summary.setText(
                f"R²       = {r2:.4f}\n"
                f"Adj R²   = {r2a:.4f}\n"
                f"RMSE     = {rmse:.4f}\n"
                f"N        = {int(model.nobs)}\n"
                f"F-stat   = {model.fvalue:.3f}  (p={model.f_pvalue:.4f})\n"
                f"AIC      = {model.aic:.2f}\n"
                f"BIC      = {model.bic:.2f}")
        except Exception as e:
            self.txt_summary.setText(f"Özet hatası: {e}")
        try:
            self.fig_analysis.clear()
            axes = self.fig_analysis.subplots(2, 2)
            self._plot_pareto(axes[0, 0], model)
            self._plot_pred_actual(axes[0, 1], model)
            self._plot_residuals_normal(axes[1, 0], model)
            self._plot_residuals_fitted(axes[1, 1], model)
            for ax in axes.flat:
                ax.set_facecolor("#0e1525")
                ax.tick_params(colors=TXT2, labelsize=8)
                for sp in ax.spines.values(): sp.set_color("#2a4060")
            self.fig_analysis.patch.set_facecolor(BG2)
            self.canvas_analysis.draw()
        except Exception as e:
            print(f"Plot error: {e}")

    def _fill_anova(self, anova):
        # mean_sq sütununu hesapla — anova_lm bunu döndürmez
        anova = anova.copy()
        anova["mean_sq"] = anova.apply(
            lambda r: r["sum_sq"]/r["df"]
            if (not math.isnan(r["sum_sq"]) and r["df"] and r["df"]>0)
            else np.nan, axis=1)

        self.anova_table.setRowCount(len(anova))
        cols = ["df", "sum_sq", "mean_sq", "F", "PR(>F)"]
        self.anova_table.setColumnCount(len(cols) + 1)
        self.anova_table.setHorizontalHeaderLabels(["Kaynak"] + cols)
        for ri, (idx, row) in enumerate(anova.iterrows()):
            self.anova_table.setItem(ri, 0, QTableWidgetItem(str(idx)))
            for ci, col in enumerate(cols):
                val = row.get(col, np.nan)
                if not isinstance(val, (int,float)): val = np.nan
                try:
                    is_nan = math.isnan(float(val))
                except: is_nan = True
                if is_nan:                txt = "—"
                elif col == "df":         txt = str(int(val))
                elif col == "PR(>F)":     txt = f"{val:.4f}"
                else:                     txt = f"{val:.4f}"
                item = QTableWidgetItem(txt)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if col == "PR(>F)" and not is_nan:
                    item.setForeground(
                        QColor("#70e870") if val < 0.05 else QColor(TXT))
                self.anova_table.setItem(ri, ci + 1, item)
        self.anova_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)

    def _plot_pareto(self, ax, model):
        pvals  = model.pvalues.drop("Intercept", errors="ignore")
        t_vals = model.tvalues.drop("Intercept", errors="ignore")
        abs_t  = t_vals.abs().sort_values(ascending=True)
        colors = ["#70e870" if pvals.get(n, 1) < 0.05
                  else "#4a6080" for n in abs_t.index]
        ax.barh(range(len(abs_t)), abs_t.values, color=colors)
        ax.set_yticks(range(len(abs_t)))
        ax.set_yticklabels([n[:18] for n in abs_t.index], fontsize=7, color=TXT)
        ax.axvline(2.0, color=GOLD, lw=1, ls="--", alpha=0.7)
        ax.set_title("Pareto — |t değerleri|", fontsize=9, color=TXT)
        ax.set_xlabel("|t|", fontsize=8, color=TXT2)

    def _plot_pred_actual(self, ax, model):
        actual = model.model.endog
        pred   = model.fittedvalues
        ax.scatter(pred, actual, color="#2E75B6", s=30, alpha=0.8, zorder=3)
        mn = min(min(actual), min(pred)); mx = max(max(actual), max(pred))
        ax.plot([mn, mx], [mn, mx], color=GOLD, lw=1.2, ls="--")
        ax.set_xlabel("Tahmin", fontsize=8, color=TXT2)
        ax.set_ylabel("Gerçek", fontsize=8, color=TXT2)
        ax.set_title("Tahmin vs Gerçek", fontsize=9, color=TXT)

    def _plot_residuals_normal(self, ax, model):
        resid = model.resid
        (osm, osr), (slope, intercept, r) = stats.probplot(resid)
        ax.plot(osm, osr, "o", color="#ED7D31", ms=5, alpha=0.8)
        ax.plot(osm, slope * np.array(osm) + intercept,
                color=GOLD, lw=1.2, ls="--")
        ax.set_title("Normal Olasılık — Artıklar", fontsize=9, color=TXT)
        ax.set_xlabel("Teorik kantil", fontsize=8, color=TXT2)
        ax.set_ylabel("Artık", fontsize=8, color=TXT2)

    def _plot_residuals_fitted(self, ax, model):
        ax.scatter(model.fittedvalues, model.resid,
                   color="#70AD47", s=30, alpha=0.8)
        ax.axhline(0, color=GOLD, lw=1.2, ls="--")
        ax.set_xlabel("Fitted", fontsize=8, color=TXT2)
        ax.set_ylabel("Artık", fontsize=8, color=TXT2)
        ax.set_title("Artıklar vs Fitted", fontsize=9, color=TXT)


# ═══════════════════════════════════════════════════════════════════════════════
# SEKME 4 — RESPONSE SURFACE
# ═══════════════════════════════════════════════════════════════════════════════
class Tab4_Surface(QWidget):
    def __init__(self, project: DoEProject, app_ref, parent=None):
        super().__init__(parent)
        self.project = project
        self.app = app_ref
        self.fixed_widgets = {}
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setSpacing(10)
        lay.setContentsMargins(14, 14, 14, 14)

        ctrl = QHBoxLayout(); ctrl.setSpacing(10)
        ctrl.addWidget(section_label("🗺  Response Surface"))
        ctrl.addStretch()
        ctrl.addWidget(QLabel("Yanıt:"))
        self.combo_resp = QComboBox(); self.combo_resp.setFixedWidth(200)
        ctrl.addWidget(self.combo_resp)
        ctrl.addWidget(QLabel("X ekseni:"))
        self.combo_x = QComboBox(); self.combo_x.setFixedWidth(130)
        ctrl.addWidget(self.combo_x)
        ctrl.addWidget(QLabel("Y ekseni:"))
        self.combo_y = QComboBox(); self.combo_y.setFixedWidth(130)
        ctrl.addWidget(self.combo_y)
        self.btn_plot = make_btn("▶ Çiz", "rgba(20,60,80,0.8)", 30)
        self.btn_plot.clicked.connect(self._plot)
        ctrl.addWidget(self.btn_plot)
        lay.addLayout(ctrl)

        self.fixed_card = card_frame()
        fl = QHBoxLayout(self.fixed_card)
        fl.setContentsMargins(10, 8, 10, 8); fl.setSpacing(12)
        fl.addWidget(QLabel("Sabit faktörler:"))
        fl.addStretch()
        lay.addWidget(self.fixed_card)

        self.fig_surf = Figure(facecolor=BG2)
        self.canvas_surf = FigureCanvas(self.fig_surf)
        lay.addWidget(self.canvas_surf, 1)

    def refresh(self):
        try:
            self.fig_surf.clear()
            self.canvas_surf.draw()
            self.combo_resp.clear()
            self.combo_x.clear()
            self.combo_y.clear()
            self.fixed_widgets.clear()
            for r in self.project.responses:
                self.combo_resp.addItem(RESPONSE_LABELS.get(r, r), r)
            for f in self.project.factors:
                if f["type"] == "continuous":
                    self.combo_x.addItem(f["name"])
                    self.combo_y.addItem(f["name"])
            if self.combo_y.count() > 1:
                self.combo_y.setCurrentIndex(1)
            self._update_fixed()
        except Exception as e:
            print(f"Tab4 refresh error: {e}")

    def _update_fixed(self):
        fl = self.fixed_card.layout()
        # Tüm eski grup widgetlarını temizle (QLabel "Sabit faktörler:" hariç)
        while fl.count() > 2:  # 0=label, 1=stretch → diğerlerini sil
            item = fl.takeAt(1)
            if item and item.widget():
                item.widget().setParent(None)
        self.fixed_widgets.clear()
        # X ve Y dışındaki sürekli faktörleri ekle
        x_name = self.combo_x.currentText()
        y_name = self.combo_y.currentText()
        added = set()
        for f in self.project.factors:
            if f["name"] in [x_name, y_name]: continue
            if f["type"] != "continuous": continue
            if f["name"] in added: continue  # mükerrer engeli
            added.add(f["name"])
            mid = f.get("mid", (f["low"] + f["high"]) / 2)
            grp = QWidget(); grp.setStyleSheet("background:transparent;")
            gl = QHBoxLayout(grp)
            gl.setContentsMargins(0,0,0,0); gl.setSpacing(4)
            gl.addWidget(QLabel(f["name"] + ":"))
            sp = QDoubleSpinBox()
            sp.setRange(f["low"], f["high"])
            sp.setValue(mid)
            sp.setDecimals(3); sp.setFixedWidth(90)
            gl.addWidget(sp)
            self.fixed_widgets[f["name"]] = sp
            fl.insertWidget(fl.count() - 1, grp)

    def _plot(self):
        resp_key = self.combo_resp.currentData()
        if not self.project.model_results.get(resp_key):
            QMessageBox.warning(
                self, "", "Önce model fit edin (Analiz sekmesi)."); return
        x_name = self.combo_x.currentText()
        y_name = self.combo_y.currentText()
        if x_name == y_name:
            QMessageBox.warning(self, "", "X ve Y aynı faktör olamaz."); return
        x_f = next((f for f in self.project.factors if f["name"]==x_name), None)
        y_f = next((f for f in self.project.factors if f["name"]==y_name), None)
        if not x_f or not y_f: return

        fixed_vals = {name: sp.value()
                      for name, sp in self.fixed_widgets.items()}

        self.btn_plot.setEnabled(False)
        self.btn_plot.setText("⏳ Hesaplanıyor...")

        self._surf_worker = SurfaceWorker(
            self.project, resp_key, x_f, y_f,
            x_name, y_name, fixed_vals, n=40)
        self._surf_worker.finished.connect(
            lambda XX,YY,ZZ: self._on_surface_ready(
                XX,YY,ZZ,resp_key,x_name,y_name))
        self._surf_worker.error.connect(self._on_surface_error)
        self._surf_worker.start()

    def _on_surface_error(self, msg):
        self.btn_plot.setEnabled(True)
        self.btn_plot.setText("▶ Çiz")
        QMessageBox.critical(self, "Tahmin Hatası", msg)

    def _on_surface_ready(self, XX, YY, ZZ, resp_key, x_name, y_name):
        self.btn_plot.setEnabled(True)
        self.btn_plot.setText("▶ Çiz")
        self.fig_surf.clear()
        ax3 = self.fig_surf.add_subplot(121, projection="3d")
        ax2 = self.fig_surf.add_subplot(122)
        self.fig_surf.patch.set_facecolor(BG2)

        ax3.plot_surface(XX, YY, ZZ, cmap="coolwarm", alpha=0.85, edgecolor="none")
        ax3.set_xlabel(x_name, fontsize=8, color=TXT2)
        ax3.set_ylabel(y_name, fontsize=8, color=TXT2)
        ax3.set_zlabel(RESPONSE_LABELS.get(resp_key,resp_key), fontsize=7, color=TXT2)
        ax3.set_title("Response Surface", fontsize=9, color=TXT)
        ax3.set_facecolor("#0e1525")
        ax3.tick_params(colors=TXT2, labelsize=7)

        cp  = ax2.contourf(XX, YY, ZZ, levels=15, cmap="coolwarm")
        cs  = ax2.contour(XX, YY, ZZ, levels=8, colors="white",
                          alpha=0.5, linewidths=0.8)
        ax2.clabel(cs, inline=True, fontsize=7, fmt="%.2f",
                   colors="white")  # ← değer etiketleri
        self.fig_surf.colorbar(cp, ax=ax2, shrink=0.8)
        ax2.set_xlabel(x_name, fontsize=8, color=TXT2)
        ax2.set_ylabel(y_name, fontsize=8, color=TXT2)
        ax2.set_title("Kontur Haritası", fontsize=9, color=TXT)
        ax2.set_facecolor("#0e1525")
        ax2.tick_params(colors=TXT2, labelsize=8)
        self.canvas_surf.draw()


# ═══════════════════════════════════════════════════════════════════════════════
# SEKME 5 — OPTİMİZASYON
# ═══════════════════════════════════════════════════════════════════════════════
class Tab5_Optimization(QWidget):
    def __init__(self, project: DoEProject, app_ref, parent=None):
        super().__init__(parent)
        self.project = project
        self.app = app_ref
        self.tgt_rows = {}
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setSpacing(10)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.addWidget(section_label("🎯  Desirability Optimizasyonu"))

        tgt_card = card_frame()
        tl = QVBoxLayout(tgt_card)
        tl.setContentsMargins(12, 10, 12, 10); tl.setSpacing(6)
        tl.addWidget(section_label("Yanıt Hedefleri"))

        # Başlık
        hdr = QWidget(); hdr.setStyleSheet("background:transparent;")
        hl = QHBoxLayout(hdr); hl.setContentsMargins(0,0,0,0); hl.setSpacing(8)
        for txt, w in [("Yanıt","160"),("Hedef","140"),("Hedef Değer (sadece 'Hedefe Ulaş')","250")]:
            l = QLabel(txt)
            l.setFixedWidth(int(w))
            l.setStyleSheet(f"color:{GOLD}; font-size:10px; font-weight:bold; background:transparent;")
            hl.addWidget(l)
        hl.addStretch()
        tl.addWidget(hdr)

        self.tgt_scroll = QScrollArea()
        self.tgt_scroll.setWidgetResizable(True)
        self.tgt_scroll.setFixedHeight(170)
        self.tgt_scroll.setStyleSheet("border:none; background:transparent;")
        self.tgt_container = QWidget()
        self.tgt_container.setStyleSheet("background:transparent;")
        self.tgt_layout = QVBoxLayout(self.tgt_container)
        self.tgt_layout.setSpacing(4)
        self.tgt_layout.setContentsMargins(0, 0, 0, 0)
        self.tgt_layout.addStretch()
        self.tgt_scroll.setWidget(self.tgt_container)
        tl.addWidget(self.tgt_scroll)
        lay.addWidget(tgt_card)

        bar = QHBoxLayout(); bar.addStretch()
        self.btn_opt = make_btn("🚀 Optimize Et", "rgba(20,80,20,0.8)", 36)
        self.btn_opt.clicked.connect(self._optimize)
        bar.addWidget(self.btn_opt)
        lay.addLayout(bar)

        res_card = card_frame()
        rl = QVBoxLayout(res_card)
        rl.setContentsMargins(12, 10, 12, 10); rl.setSpacing(8)
        rl.addWidget(section_label("📌 Optimum Formülasyon Önerisi"))
        self.txt_opt_result = QTextEdit()
        self.txt_opt_result.setReadOnly(True)
        self.txt_opt_result.setStyleSheet(f"""
            QTextEdit {{ background:{BG3}; border:1px solid #2a4060;
                         border-radius:4px; color:{TXT};
                         font-family:Consolas,monospace; font-size:12px; }}
        """)
        rl.addWidget(self.txt_opt_result)
        lay.addWidget(res_card, 1)

    def refresh(self):
        try:
            for info in self.tgt_rows.values():
                w = info.get("widget")
                if w: w.setParent(None)
        except Exception as e:
            print(f"Tab5 clear error: {e}")
        self.tgt_rows.clear()
        self.txt_opt_result.clear()
        for resp in self.project.responses:
            row = QWidget(); row.setStyleSheet("background:transparent;")
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(8)
            lbl = QLabel(RESPONSE_LABELS.get(resp, resp))
            lbl.setFixedWidth(160)
            combo = QComboBox()
            combo.addItems(["Minimize Et", "Maximize Et", "Hedefe Ulaş"])
            combo.setFixedWidth(140)
            target_sp = QDoubleSpinBox()
            target_sp.setRange(-1e6, 1e6); target_sp.setValue(0)
            target_sp.setDecimals(4); target_sp.setFixedWidth(120)
            target_sp.setEnabled(False)
            combo.currentTextChanged.connect(
                lambda t, sp=target_sp: sp.setEnabled(t == "Hedefe Ulaş"))
            rl.addWidget(lbl); rl.addWidget(combo)
            rl.addWidget(target_sp); rl.addStretch()
            self.tgt_rows[resp] = {
                "combo": combo, "target": target_sp, "widget": row}
            self.tgt_layout.insertWidget(self.tgt_layout.count() - 1, row)

    def _optimize(self):
        if not self.project.model_results:
            QMessageBox.warning(
                self, "", "Önce Analiz sekmesinden 'Model Fit' yapın."); return
        cont_idx = [i for i,f in enumerate(self.project.factors)
                    if f["type"]=="continuous"]
        if not cont_idx:
            QMessageBox.warning(self,"","Optimize edilecek sürekli faktör yok."); return

        resp_ranges = {}
        df2 = self.project.build_response_table()
        for resp in self.project.responses:
            if df2 is not None and resp in df2.columns:
                col_vals = df2[resp].dropna()
                if len(col_vals) >= 2:
                    resp_ranges[resp] = (float(col_vals.min()), float(col_vals.max()))

        self.btn_opt.setEnabled(False)
        self.btn_opt.setText("⏳ Hesaplanıyor...")
        self.txt_opt_result.setText("Optimizasyon çalışıyor, lütfen bekleyin...")

        self._opt_worker = OptimizeWorker(
            self.project, self.tgt_rows, resp_ranges)
        self._opt_worker.finished.connect(self._on_opt_done)
        self._opt_worker.error.connect(self._on_opt_error)
        self._opt_worker.start()

    def _on_opt_error(self, msg):
        self.btn_opt.setEnabled(True)
        self.btn_opt.setText("🚀 Optimize Et")
        QMessageBox.critical(self, "Optimizasyon Hatası", msg)

    def _on_opt_done(self, x_opt, d_opt):
        self.btn_opt.setEnabled(True)
        self.btn_opt.setText("🚀 Optimize Et")
        all_factors = self.project.factors
        cont_idx = [i for i,f in enumerate(all_factors) if f["type"]=="continuous"]
        actual_opt = []
        cont_ptr = 0
        for i, f in enumerate(all_factors):
            if i in cont_idx:
                actual_opt.append(x_opt[cont_ptr]); cont_ptr += 1
            else:
                actual_opt.append(f.get("mid",(f["low"]+f["high"])/2))

        lines = [
            "─" * 54,
            "   OPTİMUM FORMÜLASYON ÖNERİSİ",
            "─" * 54, "",
            "  FAKTÖR DEĞERLERİ:"
        ]
        for i, f in enumerate(all_factors):
            unit = f"  {f['unit']}" if f.get("unit") else ""
            lines.append(f"    {f['name']:28s} = {actual_opt[i]:.4f}{unit}")

        lines += ["", "  TAHMİN EDİLEN YANIT DEĞERLERİ:"]
        for resp in self.project.responses:
            lbl = RESPONSE_LABELS.get(resp, resp)
            if resp not in self.project.model_results:
                lines.append(f"    {lbl:34s} = (model yok)")
                continue
            pred, ci = self.project.predict_at(resp, actual_opt)
            if pred is None:
                lines.append(f"    {lbl:34s} = (tahmin hatası)")
            else:
                ci_str = (f"  [%95 PI: {ci[0]:.4f} – {ci[1]:.4f}]" if ci else "")
                lines.append(f"    {lbl:34s} = {pred:.4f}{ci_str}")

        d_yorum = ("Mükemmel" if d_opt>0.8 else "İyi" if d_opt>0.6
                   else "Kabul Edilebilir" if d_opt>0.4 else "Zayıf")
        lines += ["",
            f"  Genel Desirability  = {d_opt:.4f}  ({d_yorum})",
            "─" * 54]
        self.txt_opt_result.setText("\n".join(lines))




# ═══════════════════════════════════════════════════════════════════════════════
# SEKME 6 — TASARIM UZAYI
# ═══════════════════════════════════════════════════════════════════════════════
class Tab6_DesignSpace(QWidget):
    def __init__(self, project: DoEProject, app_ref, parent=None):
        super().__init__(parent)
        self.project = project
        self.app = app_ref
        self._build()

    def _build(self):
        lay = QVBoxLayout(self); lay.setSpacing(10); lay.setContentsMargins(14,14,14,14)

        # Kontrol barı
        bar = QHBoxLayout(); bar.setSpacing(10)
        bar.addWidget(section_label("🗂  Tasarım Uzayı Görselleştirme"))
        bar.addStretch()
        self.cb_heatmap = QCheckBox("Kapsama Isı Haritasını Göster")
        self.cb_heatmap.setStyleSheet("background:transparent;")
        self.cb_heatmap.stateChanged.connect(self._draw)
        bar.addWidget(self.cb_heatmap)
        self.btn_draw = make_btn("▶ Çiz", "rgba(20,60,80,0.8)", 30)
        self.btn_draw.clicked.connect(self._draw)
        bar.addWidget(self.btn_draw)
        lay.addLayout(bar)

        # Bilgi kartı
        self.info_card = card_frame()
        il = QHBoxLayout(self.info_card); il.setContentsMargins(14,8,14,8); il.setSpacing(30)
        self.lbl_design  = QLabel("Tasarım: —")
        self.lbl_runs    = QLabel("Run: —")
        self.lbl_factors = QLabel("Faktör: —")
        self.lbl_coverage= QLabel("Kapsama: —")
        for l in [self.lbl_design,self.lbl_runs,self.lbl_factors,self.lbl_coverage]:
            l.setStyleSheet(f"color:{TXT}; font-size:12px; font-weight:bold; background:transparent;")
            il.addWidget(l)
        il.addStretch()
        lay.addWidget(self.info_card)

        # Grafik alanı
        self.fig = Figure(facecolor=BG2)
        self.canvas = FigureCanvas(self.fig)
        lay.addWidget(self.canvas, 1)

    def refresh(self):
        try:
            self.fig.clear()
            self.canvas.draw()
            self.lbl_design.setText(f"Tasarım: {self.project.design_type}")
            df = self.project.design_matrix
            n = len(df) if df is not None else 0
            self.lbl_runs.setText(f"Run: {n}")
            self.lbl_factors.setText(f"Faktör: {len(self.project.factors)}")
            self.lbl_coverage.setText("Kapsama: —")
        except Exception as e:
            print(f"Tab6 refresh error: {e}")

    def _draw(self):
        df = self.project.design_matrix
        if df is None:
            QMessageBox.warning(self,"","Önce tasarım matrisi oluşturun."); return
        factors = [f for f in self.project.factors if f["type"]=="continuous"]
        k = len(factors)
        if k < 2:
            QMessageBox.warning(self,"","En az 2 sürekli faktör gerekli."); return

        show_heat = self.cb_heatmap.isChecked()

        # Tasarım noktalarını al
        pts = []
        for _, row in df.iterrows():
            pt = []
            for f in factors:
                val = row.get(f["name"], f.get("mid",(f["low"]+f["high"])/2))
                # normalize et [0,1]
                span = f["high"] - f["low"]
                norm = (val - f["low"]) / span if span != 0 else 0.5
                pt.append(norm)
            pts.append(pt)
        pts = np.array(pts)

        # Nokta tiplerini belirle (merkez, kenar, köşe)
        def classify(pt):
            n_center  = sum(1 for v in pt if abs(v-0.5) < 0.05)
            n_extreme = sum(1 for v in pt if v < 0.05 or v > 0.95)
            if n_center == len(pt): return "merkez"
            if n_extreme == len(pt): return "köşe"
            return "kenar"

        labels   = [classify(p) for p in pts]
        colors_map = {"merkez":"#70e870","kenar":"#E84040","köşe":"#7090b0"}
        colors   = [colors_map.get(l,"#2E75B6") for l in labels]

        self.fig.clear()
        self.fig.patch.set_facecolor(BG2)

        if k == 2:
            self._draw_2d(factors, pts, colors, labels, show_heat)
        elif k == 3:
            self._draw_3d(factors, pts, colors, labels, show_heat)
        else:
            self._draw_layered(factors, pts, colors, labels, show_heat)

        self.canvas.draw()

        # Kapsama hesapla
        grid_n = 10
        grid   = np.mgrid[tuple(slice(0,1,grid_n*1j) for _ in range(k))]
        grid   = grid.reshape(k,-1).T
        covered = 0
        for gp in grid:
            dists = np.linalg.norm(pts - gp, axis=1)
            if dists.min() < 0.25: covered += 1
        pct = covered / len(grid) * 100
        self.lbl_coverage.setText(f"Kapsama: %{pct:.0f}")

    def _draw_2d(self, factors, pts, colors, labels, show_heat):
        if show_heat:
            ax_heat = self.fig.add_subplot(121)
            ax_main = self.fig.add_subplot(122)
        else:
            ax_main = self.fig.add_subplot(111)

        # Ana scatter
        for lbl, col in [("köşe","#7090b0"),("kenar","#E84040"),("merkez","#70e870")]:
            idx = [i for i,l in enumerate(labels) if l==lbl]
            if idx:
                ax_main.scatter(pts[idx,0], pts[idx,1], c=col, s=120,
                               zorder=5, label=lbl.capitalize(), edgecolors="white", lw=0.5)

        # Küp çerçevesi
        for x in [0,1]:
            ax_main.axvline(x, color="#2a4060", lw=0.8, ls="--", alpha=0.5)
        for y in [0,1]:
            ax_main.axhline(y, color="#2a4060", lw=0.8, ls="--", alpha=0.5)

        # Run numaraları
        for i, (px,py) in enumerate(pts):
            ax_main.annotate(str(i+1), (px,py),
                           textcoords="offset points", xytext=(6,4),
                           fontsize=7, color=TXT2)

        ax_main.set_xlim(-0.1, 1.1); ax_main.set_ylim(-0.1, 1.1)
        ax_main.set_xlabel(factors[0]["name"], color=TXT2, fontsize=9)
        ax_main.set_ylabel(factors[1]["name"], color=TXT2, fontsize=9)
        ax_main.set_title("Tasarım Noktaları", color=TXT, fontsize=10)
        ax_main.set_facecolor("#0e1525")
        ax_main.tick_params(colors=TXT2, labelsize=8)
        for sp in ax_main.spines.values(): sp.set_color("#2a4060")
        ax_main.legend(fontsize=8, facecolor=BG3, labelcolor=TXT)

        # X/Y tick etiketleri gerçek değerlere çevir
        ticks = [0, 0.25, 0.5, 0.75, 1.0]
        for ax_obj, fi in [(ax_main, 0)]:
            f = factors[fi]
            real = [f["low"] + t*(f["high"]-f["low"]) for t in ticks]
            ax_obj.set_xticks(ticks)
            ax_obj.set_xticklabels([f"{v:.2f}" for v in real], fontsize=7, color=TXT2)
        f1 = factors[1]
        real1 = [f1["low"] + t*(f1["high"]-f1["low"]) for t in ticks]
        ax_main.set_yticks(ticks)
        ax_main.set_yticklabels([f"{v:.2f}" for v in real1], fontsize=7, color=TXT2)

        if show_heat:
            self._draw_heatmap_2d(ax_heat, pts, factors)

    def _draw_heatmap_2d(self, ax, pts, factors):
        n = 20
        xx, yy = np.mgrid[0:1:n*1j, 0:1:n*1j]
        density = np.zeros((n,n))
        for px,py in pts[:,:2]:
            ix = int(min(px*(n-1), n-1))
            iy = int(min(py*(n-1), n-1))
            density[ix,iy] += 1
        from scipy.ndimage import gaussian_filter
        density = gaussian_filter(density, sigma=1.5)
        im = ax.imshow(density.T, origin="lower", extent=[0,1,0,1],
                      cmap="YlOrRd", aspect="auto", alpha=0.85)
        self.fig.colorbar(im, ax=ax, shrink=0.8, label="Kapsama yoğunluğu")
        ax.scatter(pts[:,0], pts[:,1], c="white", s=40, zorder=5,
                  edgecolors="black", lw=0.5)
        ax.set_xlabel(factors[0]["name"], color=TXT2, fontsize=9)
        ax.set_ylabel(factors[1]["name"], color=TXT2, fontsize=9)
        ax.set_title("Kapsama Isı Haritası", color=TXT, fontsize=10)
        ax.set_facecolor("#0e1525")
        ax.tick_params(colors=TXT2, labelsize=7)
        for sp in ax.spines.values(): sp.set_color("#2a4060")

    def _draw_3d(self, factors, pts, colors, labels, show_heat):
        if show_heat:
            ax3 = self.fig.add_subplot(121, projection="3d")
            ax_h = self.fig.add_subplot(122)
        else:
            ax3 = self.fig.add_subplot(111, projection="3d")

        for lbl, col in [("köşe","#7090b0"),("kenar","#E84040"),("merkez","#70e870")]:
            idx = [i for i,l in enumerate(labels) if l==lbl]
            if idx:
                ax3.scatter(pts[idx,0], pts[idx,1], pts[idx,2],
                           c=col, s=100, zorder=5, label=lbl.capitalize(),
                           edgecolors="white", linewidths=0.5)

        # Küp tel çerçevesi
        for i in range(2):
            for j in range(2):
                ax3.plot([i,i],[j,j],[0,1], color="#2a4060", lw=0.6, alpha=0.5)
                ax3.plot([i,i],[0,1],[j,j], color="#2a4060", lw=0.6, alpha=0.5)
                ax3.plot([0,1],[i,i],[j,j], color="#2a4060", lw=0.6, alpha=0.5)

        for i, pt in enumerate(pts):
            ax3.text(pt[0], pt[1], pt[2], str(i+1),
                    fontsize=7, color=TXT2)

        ax3.set_xlabel(factors[0]["name"], color=TXT2, fontsize=8)
        ax3.set_ylabel(factors[1]["name"], color=TXT2, fontsize=8)
        ax3.set_zlabel(factors[2]["name"], color=TXT2, fontsize=8)
        ax3.set_title(f"{self.project.design_type}", color=TXT, fontsize=9)
        ax3.set_facecolor("#0e1525")
        ax3.tick_params(colors=TXT2, labelsize=7)
        ax3.legend(fontsize=8, facecolor=BG3, labelcolor=TXT)

        if show_heat:
            self._draw_heatmap_2d(ax_h, pts[:,:2], factors[:2])

    def _draw_layered(self, factors, pts, colors, labels, show_heat):
        # 4+ faktör: D faktörü katmanlar halinde göster
        # İlk 3 faktörü kullan, 4. faktör katman
        n_layers = 3
        layer_vals = np.linspace(0, 1, n_layers)
        layer_names = ["-1 (Alt)", "0 (Orta)", "+1 (Üst)"]

        cols_count = n_layers + (1 if show_heat else 0)
        axes = []
        for i in range(n_layers):
            ax = self.fig.add_subplot(1, cols_count, i+1)
            axes.append(ax)

        for li, (lv, lname) in enumerate(zip(layer_vals, layer_names)):
            ax = axes[li]
            # Bu katmana yakın noktaları filtrele
            if pts.shape[1] > 3:
                mask = np.abs(pts[:,3] - lv) < 0.35
            else:
                mask = np.ones(len(pts), dtype=bool)
            sub_pts = pts[mask]
            sub_col = [colors[i] for i,m in enumerate(mask) if m]
            sub_lbl = [labels[i] for i,m in enumerate(mask) if m]
            sub_idx = [i for i,m in enumerate(mask) if m]

            if len(sub_pts) > 0:
                ax.scatter(sub_pts[:,0], sub_pts[:,1],
                          c=sub_col, s=100, zorder=5,
                          edgecolors="white", lw=0.5)
                for j, (px, py) in enumerate(sub_pts[:,:2]):
                    ax.annotate(str(sub_idx[j]+1), (px,py),
                               textcoords="offset points", xytext=(5,3),
                               fontsize=7, color=TXT2)

            ax.set_xlim(-0.1,1.1); ax.set_ylim(-0.1,1.1)
            ax.set_xlabel(factors[0]["name"], color=TXT2, fontsize=8)
            ax.set_ylabel(factors[1]["name"] if li==0 else "", color=TXT2, fontsize=8)
            ax.set_title(f"{factors[3]['name'] if len(factors)>3 else 'F4'} = {lname}",
                        color=GOLD, fontsize=9)
            ax.set_facecolor("#0e1525")
            ax.tick_params(colors=TXT2, labelsize=7)
            for sp in ax.spines.values(): sp.set_color("#2a4060")

            # Izgara
            for v in [0, 0.5, 1]:
                ax.axvline(v, color="#2a4060", lw=0.5, ls="--", alpha=0.4)
                ax.axhline(v, color="#2a4060", lw=0.5, ls="--", alpha=0.4)

        if show_heat:
            ax_h = self.fig.add_subplot(1, cols_count, cols_count)
            self._draw_heatmap_2d(ax_h, pts[:,:2], factors[:2])

        self.fig.suptitle(
            f"{self.project.design_type} — Tasarım Uzayı",
            color=TXT, fontsize=11, y=1.01)

# ═══════════════════════════════════════════════════════════════════════════════
# ANA UYGULAMA
# ═══════════════════════════════════════════════════════════════════════════════
class DoEApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.project = DoEProject()
        self.setWindowTitle("NGI DoE Analyzer")
        self.setMinimumSize(1200, 760)
        ico_path = resource_path("doe_icon.ico")
        if os.path.exists(ico_path):
            self.setWindowIcon(QIcon(ico_path))
        self._build_ui()
        self.setStyleSheet(STYLE)

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        ml = QVBoxLayout(central)
        ml.setContentsMargins(0, 0, 0, 0); ml.setSpacing(0)

        # Başlık
        hdr = QWidget(); hdr.setFixedHeight(52)
        hdr.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 #0a1628, stop:1 #0e1f3a);")
        hl = QHBoxLayout(hdr); hl.setContentsMargins(18, 0, 18, 0); hl.setSpacing(10)
        ico_lbl = QLabel("🔬")
        ico_lbl.setStyleSheet("font-size:22px; background:transparent;")
        hl.addWidget(ico_lbl)
        t1 = QLabel("NGI DoE Analyzer")
        t1.setStyleSheet(
            "color:white; font-size:16px; font-weight:bold; background:transparent;")
        hl.addWidget(t1)
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color:#2a4060; background:#2a4060;"); sep.setFixedWidth(1)
        hl.addWidget(sep)
        t2 = QLabel("Design of Experiments — Farmasötik Formülasyon Optimizasyonu")
        t2.setStyleSheet(f"color:{TXT2}; font-size:11px; background:transparent;")
        hl.addWidget(t2)
        hl.addStretch()
        btn_pdf = make_btn("📄 PDF Rapor", "rgba(60,20,90,0.8)", 32)
        btn_pdf.clicked.connect(self.export_pdf)
        hl.addWidget(btn_pdf)
        ml.addWidget(hdr)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        ml.addWidget(self.tabs, 1)

        self.tab1 = Tab1_Factors(self.project, self)
        self.tab2 = Tab2_Design(self.project, self)
        self.tab3 = Tab3_Analysis(self.project, self)
        self.tab4 = Tab4_Surface(self.project, self)
        self.tab5 = Tab5_Optimization(self.project, self)
        self.tab6 = Tab6_DesignSpace(self.project, self)

        self.tabs.addTab(self.tab1, "1 · Faktörler & Tasarım")
        self.tabs.addTab(self.tab2, "2 · Tasarım Matrisi")
        self.tabs.addTab(self.tab3, "3 · Model & ANOVA")
        self.tabs.addTab(self.tab4, "4 · Response Surface")
        self.tabs.addTab(self.tab5, "5 · Optimizasyon")
        self.tabs.addTab(self.tab6, "6 · Tasarım Uzayı")

    def refresh_all_tabs(self):
        for tab in [self.tab2, self.tab3, self.tab4, self.tab5, self.tab6]:
            try:
                tab.refresh()
            except Exception as e:
                print(f"Tab refresh error ({tab.__class__.__name__}): {e}")

    def export_pdf(self):
        """PDF rapor oluştur"""
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, Image as RLImage,
                                        HRFlowable, PageBreak)
        import io, tempfile

        path, _ = QFileDialog.getSaveFileName(
            self, "PDF Kaydet",
            f"DoE_Rapor_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            "PDF (*.pdf)")
        if not path: return

        doc = SimpleDocTemplate(path, pagesize=A4,
            leftMargin=2*cm, rightMargin=2*cm,
            topMargin=2*cm, bottomMargin=2*cm)

        styles = getSampleStyleSheet()
        style_h1 = ParagraphStyle("H1", parent=styles["Heading1"],
            fontSize=16, textColor=colors.HexColor("#002D62"),
            spaceAfter=8)
        style_h2 = ParagraphStyle("H2", parent=styles["Heading2"],
            fontSize=12, textColor=colors.HexColor("#1F4E79"),
            spaceAfter=6, spaceBefore=12)
        style_body = ParagraphStyle("Body", parent=styles["Normal"],
            fontSize=9, spaceAfter=4, leading=14)
        style_gold = ParagraphStyle("Gold", parent=styles["Normal"],
            fontSize=9, textColor=colors.HexColor("#996600"))

        story = []
        p = self.project

        # Başlık
        story.append(Paragraph("NGI DoE Analiz Raporu", style_h1))
        story.append(HRFlowable(width="100%", thickness=2,
                                color=colors.HexColor("#FFC600")))
        story.append(Spacer(1, 0.3*cm))

        # Proje bilgisi
        story.append(Paragraph("Proje Bilgileri", style_h2))
        meta = [
            ["Ürün",      p.product   or "—"],
            ["Lot No.",   p.batch     or "—"],
            ["Analist",   p.analyst   or "—"],
            ["Tarih",     p.date      or "—"],
            ["Tasarım",   p.design_type],
            ["Run Sayısı", str(len(p.design_matrix)) if p.design_matrix is not None else "—"],
            ["Faktör Sayısı", str(len(p.factors))],
        ]
        t = Table(meta, colWidths=[4*cm, 12*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#E8F0F8")),
            ("FONTNAME",   (0,0), (0,-1), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 9),
            ("GRID",       (0,0), (-1,-1), 0.5, colors.HexColor("#CCCCCC")),
            ("ROWBACKGROUNDS", (0,0), (-1,-1),
             [colors.white, colors.HexColor("#F8F8F8")]),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.4*cm))

        # Faktörler tablosu
        story.append(Paragraph("Faktörler", style_h2))
        fdata = [["Faktör", "Tip", "Alt", "Merkez", "Üst", "Birim"]]
        for f in p.factors:
            fdata.append([
                f["name"],
                "Sürekli" if f["type"]=="continuous" else "Kategorik",
                f"{f['low']:.4f}", f"{f.get('mid',(f['low']+f['high'])/2):.4f}",
                f"{f['high']:.4f}", f.get("unit","—")])
        ft = Table(fdata, colWidths=[4.5*cm,3*cm,2*cm,2*cm,2*cm,2.5*cm])
        ft.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0), colors.HexColor("#1F4E79")),
            ("TEXTCOLOR", (0,0),(-1,0), colors.white),
            ("FONTNAME",  (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",  (0,0),(-1,-1), 9),
            ("GRID",      (0,0),(-1,-1), 0.5, colors.HexColor("#CCCCCC")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),
             [colors.white, colors.HexColor("#F8F8F8")]),
        ]))
        story.append(ft)
        story.append(Spacer(1, 0.4*cm))

        # Tasarım matrisi + sonuçlar
        if p.design_matrix is not None:
            story.append(Paragraph("Tasarım Matrisi & Ölçüm Sonuçları", style_h2))
            df = p.build_response_table()
            show_cols = [c for c in p.design_matrix.columns
                        if not c.endswith("_coded")]
            all_cols = show_cols + p.responses
            headers = [RESPONSE_LABELS.get(c,c) for c in all_cols]
            tdata = [headers]
            for ri in range(len(df)):
                row_data = []
                for col in all_cols:
                    if col in df.columns:
                        val = df.iloc[ri][col]
                        if col == "Run": row_data.append(str(int(val)))
                        elif isinstance(val,float): row_data.append(f"{val:.3f}")
                        else: row_data.append(str(val))
                    else:
                        v = p.run_results.get(ri,{}).get(col,"—")
                        row_data.append(f"{v:.3f}" if isinstance(v,float) else str(v))
                tdata.append(row_data)
            col_w = 16*cm / max(len(all_cols),1)
            mt = Table(tdata, colWidths=[col_w]*len(all_cols))
            mt.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0), colors.HexColor("#1F4E79")),
                ("TEXTCOLOR", (0,0),(-1,0), colors.white),
                ("FONTNAME",  (0,0),(-1,0), "Helvetica-Bold"),
                ("FONTSIZE",  (0,0),(-1,-1), 7),
                ("GRID",      (0,0),(-1,-1), 0.3, colors.HexColor("#CCCCCC")),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),
                 [colors.white, colors.HexColor("#F8F8F8")]),
            ]))
            story.append(mt)

        # Model sonuçları
        if p.model_results:
            story.append(PageBreak())
            story.append(Paragraph("Model Sonuçları", style_h2))
            for resp, model in p.model_results.items():
                lbl = RESPONSE_LABELS.get(resp,resp)
                story.append(Paragraph(f"<b>{lbl}</b>", style_gold))
                try:
                    r2 = model.rsquared; r2a = model.rsquared_adj
                    rmse = float(np.sqrt(model.mse_resid)) if model.mse_resid else 0
                    summary_data = [
                        ["R²", f"{r2:.4f}", "Adj R²", f"{r2a:.4f}"],
                        ["RMSE", f"{rmse:.4f}", "N", str(int(model.nobs))],
                        ["F-stat", f"{model.fvalue:.3f}", "p", f"{model.f_pvalue:.4f}"],
                    ]
                    st = Table(summary_data, colWidths=[3*cm,3*cm,3*cm,3*cm])
                    st.setStyle(TableStyle([
                        ("FONTSIZE",(0,0),(-1,-1),8),
                        ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
                        ("BACKGROUND",(0,0),(0,-1),colors.HexColor("#E8F0F8")),
                        ("BACKGROUND",(2,0),(2,-1),colors.HexColor("#E8F0F8")),
                    ]))
                    story.append(st)
                except: pass
                story.append(Spacer(1, 0.2*cm))

                # ANOVA
                try:
                    from statsmodels.stats.anova import anova_lm
                    anova = anova_lm(model, typ=2)
                    anova_data = [["Kaynak","df","SS","MS","F","p"]]
                    for idx2, row2 in anova.iterrows():
                        df_v = row2.get("df",np.nan)
                        ss_v = row2.get("sum_sq",np.nan)
                        ms_v = ss_v/df_v if (not math.isnan(ss_v) and df_v and df_v>0) else np.nan
                        f_v  = row2.get("F",np.nan)
                        p_v  = row2.get("PR(>F)",np.nan)
                        def fmt(v): return "—" if isinstance(v,float) and math.isnan(v) else f"{v:.4f}"
                        anova_data.append([
                            str(idx2),
                            str(int(df_v)) if not math.isnan(df_v) else "—",
                            fmt(ss_v), fmt(ms_v), fmt(f_v), fmt(p_v)])
                    at = Table(anova_data,
                              colWidths=[5*cm,1.5*cm,2*cm,2*cm,2*cm,2*cm])
                    at.setStyle(TableStyle([
                        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1F4E79")),
                        ("TEXTCOLOR", (0,0),(-1,0),colors.white),
                        ("FONTNAME",  (0,0),(-1,0),"Helvetica-Bold"),
                        ("FONTSIZE",  (0,0),(-1,-1),8),
                        ("GRID",      (0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
                        ("ROWBACKGROUNDS",(0,1),(-1,-1),
                         [colors.white,colors.HexColor("#F8F8F8")]),
                    ]))
                    story.append(at)
                except: pass
                story.append(Spacer(1, 0.4*cm))

        # Optimizasyon sonucu
        opt_text = self.tab5.txt_opt_result.toPlainText()
        if opt_text and "─" in opt_text:
            story.append(PageBreak())
            story.append(Paragraph("Optimizasyon Sonucu", style_h2))
            for line in opt_text.split("\n"):
                story.append(Paragraph(
                    line.replace(" ","&nbsp;"),
                    ParagraphStyle("Mono", parent=styles["Normal"],
                        fontSize=8, fontName="Courier", leading=12)))

        # Grafikleri kaydet
        story.append(PageBreak())
        story.append(Paragraph("Grafikler", style_h2))

        for fig_obj, title in [
            (self.tab3.fig_analysis, "Model Analiz Grafikleri"),
            (self.tab4.fig_surf,     "Response Surface"),
            (self.tab6.fig,          "Tasarım Uzayı"),
        ]:
            try:
                buf = io.BytesIO()
                fig_obj.savefig(buf, format="png", dpi=120,
                               facecolor=fig_obj.get_facecolor(),
                               bbox_inches="tight")
                buf.seek(0)
                img = RLImage(buf, width=16*cm, height=10*cm)
                story.append(Paragraph(f"<b>{title}</b>", style_gold))
                story.append(img)
                story.append(Spacer(1, 0.4*cm))
            except Exception as e:
                print(f"Grafik PDF hatası ({title}): {e}")

        try:
            doc.build(story)
            import subprocess, platform
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
            QMessageBox.information(self, "✔", f"PDF oluşturuldu:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "PDF Hatası", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("NGI DoE Analyzer")
    win = DoEApp()
    win.show()
    sys.exit(app.exec())
