"""
NGI DoE Analyzer
Design of Experiments için NGI-CITDAS entegre analiz aracı
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

# ─── resource_path ────────────────────────────────────────────────────────────
def resource_path(rel):
    base = getattr(sys, '_MEIPASS',
        os.path.dirname(os.path.abspath(
            sys.executable if getattr(sys,'frozen',False) else __file__)))
    return os.path.join(base, rel)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
from PyQt6.QtCore import Qt, QSize, QTimer
from PyQt6.QtGui import QColor, QFont, QIcon, QPalette

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
    "mmad":       "MMAD (µm)",
    "gsd":        "GSD",
    "fpd_5um":    "FPD <5µm (mg)",
    "fpf_5um":    "FPF <5µm (%)",
    "fpd_3um":    "FPD <3µm (mg)",
    "fpf_3um":    "FPF <3µm (%)",
    "fpd_15um":   "FPD <1.5µm (mg)",
    "fpf_15um":   "FPF <1.5µm (%)",
    "metered":    "Metered Doz (mg)",
    "delivered":  "Delivered Doz (mg)",
}

DESIGN_TYPES = [
    "Full Factorial (2k)",
    "Fractional Factorial (2k-p)",
    "Central Composite (CCD/RSM)",
    "Box-Behnken (BBD)",
    "Plackett-Burman",
    "One Factor at a Time (OFAT)",
]

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
# VERİ MODELİ
# ═══════════════════════════════════════════════════════════════════════════════
class DoEProject:
    """Tüm proje verisini tutan merkezi model"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.ngi_data      = None   # JSON'dan yüklenen ham NGI verisi
        self.factors       = []     # [{"name","type","low","high","unit","levels"}]
        self.responses     = []     # ["mmad","fpf_5um", ...]
        self.design_type   = DESIGN_TYPES[0]
        self.design_matrix = None   # pd.DataFrame — run x faktör
        self.run_results   = {}     # {run_idx: {resp: value}}
        self.model_results = {}     # {resp: statsmodels result}
        self.product       = ""
        self.batch         = ""
        self.analyst       = ""
        self.date          = ""
        self.flow_rate     = 60
        self.notes         = ""

    def get_factor_names(self):
        return [f["name"] for f in self.factors]

    def build_response_table(self):
        """Design matrix + sonuçları birleştir"""
        if self.design_matrix is None: return None
        df = self.design_matrix.copy()
        for resp in self.responses:
            df[resp] = [self.run_results.get(i, {}).get(resp, np.nan)
                        for i in range(len(df))]
        return df

    def get_safe_names(self):
        """Faktör adlarını model-güvenli isimlere çevir"""
        return [f"X{i+1}" for i in range(len(self.factors))]

    def get_coded_df(self):
        """Coded değerler (-1/+1) üzerinden model DataFrame döndür"""
        df = self.design_matrix
        if df is None: return None, []
        safe_names = self.get_safe_names()
        # Coded sütunları al (varsa), yoksa actual'dan hesapla
        coded_cols = {}
        for i, f in enumerate(self.factors):
            sn = safe_names[i]
            coded_col = f["name"] + "_coded"
            if coded_col in df.columns:
                coded_cols[sn] = df[coded_col].values
            elif f["type"] == "continuous" and f["high"] != f["low"]:
                actual = df[f["name"]].values
                coded = 2 * (actual - f["low"]) / (f["high"] - f["low"]) - 1
                coded_cols[sn] = coded
            else:
                coded_cols[sn] = df[f["name"]].values
        base_df = pd.DataFrame(coded_cols)
        return base_df, safe_names

    def fit_models(self):
        """Tüm yanıtlar için OLS model fit et (coded değerler üzerinden)"""
        base_df, safe_names = self.get_coded_df()
        if base_df is None: return
        self.model_results = {}
        self.model_errors = {}
        for resp in self.responses:
            # Yanıt değerlerini ekle
            y_vals = [self.run_results.get(i, {}).get(resp, np.nan)
                      for i in range(len(base_df))]
            sub = base_df.copy()
            sub[resp] = y_vals
            sub = sub.dropna()
            n_needed = len(safe_names) + 2
            if len(sub) < n_needed:
                self.model_errors[resp] = f"Yetersiz veri: {len(sub)} run, en az {n_needed} gerekli"
                continue
            try:
                terms = safe_names[:]
                # 2-yönlü interaksiyonlar
                for a, b in itertools.combinations(safe_names, 2):
                    col = f"{a}_x_{b}"
                    sub[col] = sub[a] * sub[b]
                    terms.append(col)
                # CCD/BBD quadratic terimler
                if self.design_type in ["Central Composite (CCD/RSM)", "Box-Behnken (BBD)"]:
                    for a in safe_names:
                        col = f"{a}_sq"
                        sub[col] = sub[a] ** 2
                        terms.append(col)
                # Multicollinearity veya rank sorununa karşı: sadece anlamlı terimleri tut
                formula = f"{resp} ~ " + " + ".join(terms)
                model = ols(formula, data=sub).fit()
                self.model_results[resp] = model
            except Exception as e:
                self.model_errors[resp] = str(e)
                print(f"Model fit error ({resp}): {e}")

    def predict_at(self, resp, factor_actual_values):
        """Verilen actual faktör değerlerinde tahmin yap"""
        model = self.model_results.get(resp)
        if not model: return None, None
        safe_names = self.get_safe_names()
        row = {}
        for i, f in enumerate(self.factors):
            sn = safe_names[i]
            actual = factor_actual_values[i]
            if f["type"] == "continuous" and f["high"] != f["low"]:
                coded = 2 * (actual - f["low"]) / (f["high"] - f["low"]) - 1
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
            # %95 güven aralığı
            try:
                pi = model.get_prediction(pred_df).summary_frame(alpha=0.05)
                lo = float(pi["obs_ci_lower"].iloc[0])
                hi = float(pi["obs_ci_upper"].iloc[0])
            except:
                se = float(np.sqrt(model.mse_resid)) if model.mse_resid else 0
                lo, hi = pred - 1.96*se, pred + 1.96*se
            return pred, (lo, hi)
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
        self.setStyleSheet("background: transparent;")
        lay = QHBoxLayout(self); lay.setContentsMargins(0,2,0,2); lay.setSpacing(6)

        # Index etiketi
        lbl = QLabel(f"{idx+1}.")
        lbl.setFixedWidth(20)
        lbl.setStyleSheet(f"color:{TXT2}; font-size:11px; background:transparent;")
        lay.addWidget(lbl)

        # Faktör adı
        self.e_name = QLineEdit(f"Faktör {idx+1}")
        self.e_name.setFixedWidth(130)
        self.e_name.setPlaceholderText("Faktör adı")
        lay.addWidget(self.e_name)

        # Tip
        self.combo_type = QComboBox()
        self.combo_type.addItems(["Sürekli", "Kategorik"])
        self.combo_type.setFixedWidth(90)
        self.combo_type.currentTextChanged.connect(self._on_type_change)
        lay.addWidget(self.combo_type)

        # Alt seviye
        self.e_low = QLineEdit("0")
        self.e_low.setFixedWidth(70)
        self.e_low.setPlaceholderText("Alt")
        lay.addWidget(self.e_low)

        # Üst seviye
        self.e_high = QLineEdit("1")
        self.e_high.setFixedWidth(70)
        self.e_high.setPlaceholderText("Üst")
        lay.addWidget(self.e_high)

        # Birim
        self.e_unit = QLineEdit("")
        self.e_unit.setFixedWidth(60)
        self.e_unit.setPlaceholderText("Birim")
        lay.addWidget(self.e_unit)

        # Seviye sayısı (kategorik için)
        self.spin_levels = QSpinBox()
        self.spin_levels.setRange(2,5)
        self.spin_levels.setValue(2)
        self.spin_levels.setFixedWidth(50)
        self.spin_levels.setVisible(False)
        lay.addWidget(self.spin_levels)

        # Sil butonu
        self.btn_del = QPushButton("✕")
        self.btn_del.setFixedSize(26,26)
        self.btn_del.setStyleSheet("""
            QPushButton { background: rgba(100,20,20,0.6); border: 1px solid #6a2020;
                          border-radius: 4px; color: #ff8080; font-weight: bold; }
            QPushButton:hover { background: rgba(160,30,30,0.8); }
        """)
        lay.addWidget(self.btn_del)
        lay.addStretch()

    def _on_type_change(self, t):
        is_cont = (t == "Sürekli")
        self.e_low.setVisible(is_cont)
        self.e_high.setVisible(is_cont)
        self.spin_levels.setVisible(not is_cont)

    def get_data(self):
        is_cont = self.combo_type.currentText() == "Sürekli"
        try: low = float(self.e_low.text())
        except: low = 0.0
        try: high = float(self.e_high.text())
        except: high = 1.0
        return {
            "name":   self.e_name.text().strip() or f"F{self.idx+1}",
            "type":   "continuous" if is_cont else "categorical",
            "low":    low,
            "high":   high,
            "unit":   self.e_unit.text().strip(),
            "levels": self.spin_levels.value()
        }

    def set_data(self, d):
        self.e_name.setText(d.get("name",""))
        t = "Sürekli" if d.get("type","continuous")=="continuous" else "Kategorik"
        self.combo_type.setCurrentText(t)
        self.e_low.setText(str(d.get("low",0)))
        self.e_high.setText(str(d.get("high",1)))
        self.e_unit.setText(d.get("unit",""))
        self.spin_levels.setValue(d.get("levels",2))


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
        main = QVBoxLayout(self); main.setSpacing(12); main.setContentsMargins(14,14,14,14)

        # ── Üst: Import + Proje Bilgisi ───────────────────────────────────────
        top_row = QHBoxLayout(); top_row.setSpacing(12)

        # Import kartı
        import_card = card_frame()
        il = QVBoxLayout(import_card); il.setContentsMargins(12,10,12,10); il.setSpacing(8)
        il.addWidget(section_label("📂  NGI-CITDAS Verisi"))
        self.lbl_import_status = QLabel("Henüz veri yüklenmedi")
        self.lbl_import_status.setStyleSheet(f"color:{TXT2}; font-size:11px; background:transparent;")
        self.lbl_import_status.setWordWrap(True)
        il.addWidget(self.lbl_import_status)
        btn_row = QHBoxLayout(); btn_row.setSpacing(6)
        self.btn_import = make_btn("📂  JSON İmport", "rgba(10,60,50,0.8)")
        self.btn_import.clicked.connect(self._import_json)
        btn_row.addWidget(self.btn_import)
        self.btn_load_series = make_btn("↙  Serileri Faktöre Yükle", "rgba(20,40,80,0.8)")
        self.btn_load_series.clicked.connect(self._load_series_as_factors)
        self.btn_load_series.setEnabled(False)
        btn_row.addWidget(self.btn_load_series)
        il.addLayout(btn_row)
        top_row.addWidget(import_card, 1)

        # Proje bilgi kartı
        info_card = card_frame()
        fl = QFormLayout(info_card); fl.setContentsMargins(12,10,12,10); fl.setSpacing(6)
        fl.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        lbl_style = f"color:{TXT2}; font-size:11px; background:transparent;"
        def qlbl(t): l=QLabel(t); l.setStyleSheet(lbl_style); return l
        self.e_product  = QLineEdit(); self.e_product.setPlaceholderText("Ürün adı")
        self.e_batch    = QLineEdit(); self.e_batch.setPlaceholderText("Lot No.")
        self.e_analyst  = QLineEdit(); self.e_analyst.setPlaceholderText("Analist")
        self.e_date     = QLineEdit(datetime.date.today().strftime("%d.%m.%Y"))
        fl.addRow(qlbl("Ürün:"), self.e_product)
        fl.addRow(qlbl("Lot:"), self.e_batch)
        fl.addRow(qlbl("Analist:"), self.e_analyst)
        fl.addRow(qlbl("Tarih:"), self.e_date)
        top_row.addWidget(info_card, 1)
        main.addLayout(top_row)

        # ── Tasarım Tipi ──────────────────────────────────────────────────────
        design_card = card_frame()
        dl = QHBoxLayout(design_card); dl.setContentsMargins(12,10,12,10); dl.setSpacing(16)
        dl.addWidget(section_label("⚗️  Tasarım Tipi:"))
        self.combo_design = QComboBox()
        self.combo_design.addItems(DESIGN_TYPES)
        self.combo_design.setFixedWidth(280)
        dl.addWidget(self.combo_design)
        self.lbl_design_info = QLabel()
        self.lbl_design_info.setStyleSheet(f"color:{TXT2}; font-size:11px; background:transparent;")
        dl.addWidget(self.lbl_design_info, 1)
        self.combo_design.currentTextChanged.connect(self._update_design_info)
        dl.addStretch()
        main.addWidget(design_card)
        self._update_design_info(self.combo_design.currentText())

        # ── Faktörler ─────────────────────────────────────────────────────────
        factor_card = card_frame()
        fcl = QVBoxLayout(factor_card); fcl.setContentsMargins(12,10,12,10); fcl.setSpacing(6)
        hdr = QHBoxLayout()
        hdr.addWidget(section_label("🔬  Faktörler (Bağımsız Değişkenler)"))
        hdr.addStretch()
        btn_add_f = make_btn("＋ Faktör Ekle", "rgba(20,50,100,0.7)", 28)
        btn_add_f.clicked.connect(self._add_factor_row)
        hdr.addWidget(btn_add_f)
        fcl.addLayout(hdr)

        # Başlık satırı
        hdr_row = QWidget(); hdr_row.setStyleSheet("background:transparent;")
        hr = QHBoxLayout(hdr_row); hr.setContentsMargins(0,0,0,0); hr.setSpacing(6)
        for txt, w in [("#","20"),("Faktör Adı","130"),("Tip","90"),
                       ("Alt Seviye","70"),("Üst Seviye","70"),("Birim","60")]:
            l = QLabel(txt); l.setStyleSheet(f"color:{GOLD}; font-size:10px; font-weight:bold; background:transparent;")
            l.setFixedWidth(int(w)); hr.addWidget(l)
        hr.addStretch()
        fcl.addWidget(hdr_row)

        # Faktör listesi scroll
        self.factor_scroll = QScrollArea()
        self.factor_scroll.setWidgetResizable(True)
        self.factor_scroll.setFixedHeight(200)
        self.factor_scroll.setStyleSheet("background:transparent; border:none;")
        self.factor_container = QWidget()
        self.factor_container.setStyleSheet("background:transparent;")
        self.factor_layout = QVBoxLayout(self.factor_container)
        self.factor_layout.setSpacing(2); self.factor_layout.setContentsMargins(0,0,0,0)
        self.factor_layout.addStretch()
        self.factor_scroll.setWidget(self.factor_container)
        fcl.addWidget(self.factor_scroll)
        main.addWidget(factor_card)

        # ── Yanıt Değişkenleri ────────────────────────────────────────────────
        resp_card = card_frame()
        rcl = QVBoxLayout(resp_card); rcl.setContentsMargins(12,10,12,10); rcl.setSpacing(8)
        rcl.addWidget(section_label("🎯  Yanıt Değişkenleri (NGI Parametreleri)"))
        self.resp_checks = {}
        resp_grid = QGridLayout(); resp_grid.setSpacing(6)
        for i, (key, lbl) in enumerate(RESPONSE_LABELS.items()):
            cb = QCheckBox(lbl)
            cb.setStyleSheet("background: transparent;")
            self.resp_checks[key] = cb
            resp_grid.addWidget(cb, i//4, i%4)
        rcl.addLayout(resp_grid)
        main.addWidget(resp_card)

        # ── Alt Butonlar ──────────────────────────────────────────────────────
        btn_bar = QHBoxLayout(); btn_bar.setSpacing(8)
        btn_bar.addStretch()
        self.btn_build = make_btn("⚗️  Tasarım Matrisini Oluştur", "rgba(20,80,20,0.8)", 36)
        self.btn_build.clicked.connect(self._build_design)
        btn_bar.addWidget(self.btn_build)
        main.addLayout(btn_bar)
        main.addStretch()

        # Başlangıçta 2 faktör ekle
        self._add_factor_row(); self._add_factor_row()

    def _update_design_info(self, design):
        infos = {
            "Full Factorial (2k)":          "Her faktörün tüm kombinasyonları. Az faktörde ideal (≤5).",
            "Fractional Factorial (2k-p)":  "Tam faktöryelin alt kümesi. Tarama çalışmaları için.",
            "Central Composite (CCD/RSM)":  "RSM için. Merkez noktalar + yıldız noktaları. Quadratic model.",
            "Box-Behnken (BBD)":            "RSM için. 3 seviyeli, CCD'den az run. Extreme noktalar yok.",
            "Plackett-Burman":              "Çok faktörü hızlıca tara. N=run sayısı (4'ün katı).",
            "One Factor at a Time (OFAT)":  "Bir faktörü değiştir, diğerlerini sabit tut. Baseline.",
        }
        self.lbl_design_info.setText(infos.get(design, ""))

    def _add_factor_row(self):
        row = FactorRow(len(self.factor_rows))
        row.btn_del.clicked.connect(lambda _, r=row: self._del_factor_row(r))
        self.factor_rows.append(row)
        # stretch'ten önce ekle
        self.factor_layout.insertWidget(self.factor_layout.count()-1, row)

    def _del_factor_row(self, row):
        if len(self.factor_rows) <= 1:
            QMessageBox.warning(self, "", "En az 1 faktör olmalı."); return
        self.factor_rows.remove(row)
        row.setParent(None)
        # idx güncelle
        for i, r in enumerate(self.factor_rows): r.idx = i

    def load_json(self, path):
        """Dışarıdan (NGI-CITDAS) JSON path ile çağrılır"""
        if not path or not os.path.exists(path): return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self.project.ngi_data = data
            info = data.get("export_info", {})
            self.e_product.setText(info.get("product",""))
            self.e_batch.setText(info.get("batch",""))
            self.e_analyst.setText(info.get("operator",""))
            self.project.flow_rate = info.get("flow_rate", 60)
            n_series = len(data.get("series", []))
            self.lbl_import_status.setText(
                f"✔  {os.path.basename(path)}\n"
                f"   {n_series} seri | Flow: {self.project.flow_rate} L/min | "
                f"Ürün: {info.get('product','?')}")
            self.lbl_import_status.setStyleSheet(
                f"color: #70e870; font-size:11px; background:transparent;")
            self.btn_load_series.setEnabled(True)
            # Yanıt değişkenlerini otomatik seç
            for key, cb in self.resp_checks.items():
                cb.setChecked(key in ["mmad","fpf_5um","fpd_5um"])
        except Exception as e:
            QMessageBox.critical(self, "JSON Hata", str(e))

    def _import_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "NGI DoE Verisi Aç", "", "JSON (*.json)")
        if not path: return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self.project.ngi_data = data
            info = data.get("export_info", {})
            self.e_product.setText(info.get("product",""))
            self.e_batch.setText(info.get("batch",""))
            self.e_analyst.setText(info.get("operator",""))
            self.project.flow_rate = info.get("flow_rate", 60)
            n_series = len(data.get("series", []))
            self.lbl_import_status.setText(
                f"✔  {os.path.basename(path)}\n"
                f"   {n_series} seri | Flow: {self.project.flow_rate} L/min | "
                f"Ürün: {info.get('product','?')}")
            self.lbl_import_status.setStyleSheet(
                f"color: #70e870; font-size:11px; background:transparent;")
            self.btn_load_series.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "JSON Hata", str(e))

    def _load_series_as_factors(self):
        """NGI serilerini doğrudan faktör olarak yükle (yanıt değerleriyle)"""
        if not self.project.ngi_data: return
        series = self.project.ngi_data.get("series", [])
        if not series: return
        # Mevcut faktörleri temizle
        for row in self.factor_rows[:]:
            row.setParent(None)
        self.factor_rows.clear()
        # Her seriyi bir faktör olarak göster
        QMessageBox.information(self, "Seri Yükleme",
            f"{len(series)} seri bulundu.\n\n"
            "NGI serileri, faktörlerinizi temsil etmez — formülasyon değişkenlerinizi "
            "(viskozite, konsantrasyon vb.) manuel girmeniz gerekir.\n\n"
            "Seri verileri 'Veri Girişi' sekmesinde yanıt değerleri olarak görünecek.")
        # Varsayılan 2 faktör ekle
        self._add_factor_row(); self._add_factor_row()
        # Yanıt değişkenlerini otomatik seç
        for key, cb in self.resp_checks.items():
            cb.setChecked(key in ["mmad","fpf_5um","fpd_5um"])

    def _build_design(self):
        """Faktör ve tasarım tipine göre matrix oluştur"""
        factors = [r.get_data() for r in self.factor_rows]
        if not factors:
            QMessageBox.warning(self,"","Faktör ekleyin."); return
        responses = [k for k, cb in self.resp_checks.items() if cb.isChecked()]
        if not responses:
            QMessageBox.warning(self,"","En az 1 yanıt değişkeni seçin."); return

        design = self.combo_design.currentText()
        k = len(factors)

        try:
            matrix = self._generate_matrix(design, factors)
        except Exception as e:
            QMessageBox.critical(self,"Tasarım Hatası", str(e)); return

        # DataFrame oluştur
        cols = [f["name"] for f in factors]
        df = pd.DataFrame(matrix, columns=cols)

        # Coded → actual değerlere dönüştür (sürekli faktörler için)
        for f in factors:
            if f["type"] == "continuous" and f["name"] in df.columns:
                col = df[f["name"]]
                # Coded: -1 → low, +1 → high
                actual = f["low"] + (col + 1) / 2 * (f["high"] - f["low"])
                df[f["name"]+"_coded"] = col
                df[f["name"]] = actual.round(4)

        # Run numarası ekle
        df.insert(0, "Run", range(1, len(df)+1))

        # Projeye kaydet
        self.project.factors = factors
        self.project.responses = responses
        self.project.design_type = design
        self.project.design_matrix = df
        self.project.product  = self.e_product.text()
        self.project.batch    = self.e_batch.text()
        self.project.analyst  = self.e_analyst.text()
        self.project.date     = self.e_date.text()

        # Diğer sekmeleri güncelle
        self.app.refresh_all_tabs()
        self.app.tabs.setCurrentIndex(1)
        QMessageBox.information(self,"✔ Tasarım Oluşturuldu",
            f"{design}\n{len(df)} run | {k} faktör | {len(responses)} yanıt\n\n"
            "Tasarım Matrisi sekmesine geçildi.")

    def _generate_matrix(self, design, factors):
        k = len(factors)
        cont_factors = [f for f in factors if f["type"]=="continuous"]
        kc = len(cont_factors)

        if design == "Full Factorial (2k)":
            return pyDOE3.ff2n(k)

        elif design == "Fractional Factorial (2k-p)":
            # En uygun resolution'ı otomatik seç
            if k <= 2:   return pyDOE3.ff2n(k)
            elif k == 3: gen = "a b ab"
            elif k == 4: gen = "a b c abc"
            elif k == 5: gen = "a b c d abcd"
            else:        gen = None
            if gen:
                return pyDOE3.fracfact(gen)
            else:
                return pyDOE3.ff2n(k)  # fallback

        elif design == "Central Composite (CCD/RSM)":
            if kc < 2: raise ValueError("CCD için en az 2 sürekli faktör gerekli.")
            return pyDOE3.ccdesign(kc, center=(4,4), face="circumscribed")

        elif design == "Box-Behnken (BBD)":
            if kc < 3: raise ValueError("Box-Behnken için en az 3 sürekli faktör gerekli.")
            return pyDOE3.bbdesign(kc, center=3)

        elif design == "Plackett-Burman":
            n_runs = max(8, ((k+1)//4+1)*4)  # 4'ün katı, k+1'den büyük
            return pyDOE3.pbdesign(n_runs)[:, :k]

        elif design == "One Factor at a Time (OFAT)":
            # Merkez noktası + her faktörü alt/üst oynatan runlar
            center = np.zeros((1, k))
            runs = [center]
            for i in range(k):
                lo_run = np.zeros((1,k)); lo_run[0,i] = -1
                hi_run = np.zeros((1,k)); hi_run[0,i] =  1
                runs += [lo_run, hi_run]
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
        self._build()

    def _build(self):
        lay = QVBoxLayout(self); lay.setSpacing(10); lay.setContentsMargins(14,14,14,14)

        # Üst butonlar
        bar = QHBoxLayout(); bar.setSpacing(8)
        bar.addWidget(section_label("📋  Tasarım Matrisi & Yanıt Değerleri"))
        bar.addStretch()
        self.btn_randomize = make_btn("🔀 Randomize", "rgba(60,40,10,0.8)", 28)
        self.btn_randomize.clicked.connect(self._randomize)
        bar.addWidget(self.btn_randomize)
        self.btn_export_matrix = make_btn("⬇ Excel'e Aktar", "rgba(10,60,20,0.8)", 28)
        self.btn_export_matrix.clicked.connect(self._export_excel)
        bar.addWidget(self.btn_export_matrix)
        self.btn_autofill = make_btn("⚡ NGI Veriden Doldur", "rgba(20,40,80,0.8)", 28)
        self.btn_autofill.clicked.connect(self._autofill_from_ngi)
        bar.addWidget(self.btn_autofill)
        lay.addLayout(bar)

        # Bilgi kartı
        self.lbl_info = QLabel("Tasarım matrisi henüz oluşturulmadı. Faktörler sekmesinden başlayın.")
        self.lbl_info.setStyleSheet(f"color:{TXT2}; font-size:11px; background:transparent;")
        lay.addWidget(self.lbl_info)

        # Ana tablo
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(self.table.styleSheet() + """
            QTableWidget { alternate-background-color: #181f2e; }
        """)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.cellChanged.connect(self._on_cell_changed)
        lay.addWidget(self.table, 1)

        # NGI seri eşleştirme
        map_card = card_frame()
        ml = QVBoxLayout(map_card); ml.setContentsMargins(12,8,12,8); ml.setSpacing(6)
        ml.addWidget(section_label("🔗  Run ↔ NGI Seri Eşleştirme"))
        self.lbl_map_hint = QLabel(
            "Her run için hangi NGI serisinin yanıt değerini kullanmak istediğinizi seçin.")
        self.lbl_map_hint.setStyleSheet(f"color:{TXT2}; font-size:11px; background:transparent;")
        ml.addWidget(self.lbl_map_hint)
        self.map_scroll = QScrollArea()
        self.map_scroll.setWidgetResizable(True); self.map_scroll.setFixedHeight(110)
        self.map_scroll.setStyleSheet("background:transparent; border:none;")
        self.map_container = QWidget(); self.map_container.setStyleSheet("background:transparent;")
        self.map_layout = QHBoxLayout(self.map_container)
        self.map_layout.setSpacing(8); self.map_layout.addStretch()
        self.map_scroll.setWidget(self.map_container)
        ml.addWidget(self.map_scroll)
        lay.addWidget(map_card)

        self._col_blocking = False

    def refresh(self):
        df = self.project.design_matrix
        if df is None:
            self.lbl_info.setText("Tasarım matrisi henüz oluşturulmadı.")
            self.table.setRowCount(0); self.table.setColumnCount(0); return

        n_runs = len(df)
        responses = self.project.responses
        factor_names = self.project.get_factor_names()

        # Sadece actual değer sütunlarını göster (coded hariç)
        show_cols = [c for c in df.columns if not c.endswith("_coded")]
        all_cols = show_cols + responses

        self.lbl_info.setText(
            f"Tasarım: {self.project.design_type}  |  "
            f"{n_runs} run  |  {len(factor_names)} faktör  |  "
            f"{len(responses)} yanıt değişkeni")

        self._col_blocking = True
        self.table.setColumnCount(len(all_cols))
        self.table.setRowCount(n_runs)
        self.table.setHorizontalHeaderLabels([
            RESPONSE_LABELS.get(c, c) for c in all_cols])

        for ri in range(n_runs):
            for ci, col in enumerate(all_cols):
                if col in df.columns:
                    val = df.iloc[ri][col]
                    item = QTableWidgetItem(
                        str(int(val)) if col=="Run" else f"{val:.4f}" if isinstance(val,float) else str(val))
                    if col in show_cols:
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                        item.setForeground(QColor(TXT2))
                    else:
                        item.setForeground(QColor(GOLD))
                    self.table.setItem(ri, ci, item)
                else:
                    # Yanıt sütunu — kullanıcı girer
                    val = self.project.run_results.get(ri, {}).get(col, "")
                    item = QTableWidgetItem("" if val=="" or (isinstance(val,float) and math.isnan(val)) else f"{val:.4f}")
                    item.setForeground(QColor(GOLD))
                    self.table.setItem(ri, ci, item)

        self._col_blocking = False
        self._rebuild_map()

    def _rebuild_map(self):
        # Eski widget'ları temizle
        for i in reversed(range(self.map_layout.count()-1)):
            w = self.map_layout.itemAt(i).widget()
            if w: w.setParent(None)
        self.run_series_combos = {}
        if not self.project.design_matrix is None:
            ngi = self.project.ngi_data
            series_names = ["(Manuel)"] + (
                [s["series_id"] for s in ngi.get("series",[])] if ngi else [])
            for ri in range(min(len(self.project.design_matrix), 20)):
                grp = QGroupBox(f"Run {ri+1}")
                grp.setStyleSheet(f"""
                    QGroupBox {{ border:1px solid #2a4060; border-radius:4px;
                                 margin-top:8px; padding-top:4px; color:{TXT2}; font-size:10px; }}
                    QGroupBox::title {{ color:{GOLD}; left:6px; }}
                """)
                gl = QVBoxLayout(grp); gl.setContentsMargins(4,4,4,4)
                combo = QComboBox()
                combo.addItems(series_names)
                combo.setFixedWidth(130)
                combo.setStyleSheet(f"""
                    QComboBox {{ background:{BG3}; border:1px solid #2a4060;
                                 border-radius:3px; padding:2px 4px; color:{TXT}; font-size:10px; }}
                """)
                self.run_series_combos[ri] = combo
                gl.addWidget(combo)
                self.map_layout.insertWidget(self.map_layout.count()-1, grp)

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
            val = float(item.text().replace(",","."))
            if row not in self.project.run_results:
                self.project.run_results[row] = {}
            self.project.run_results[row][resp_key] = val
        except: pass

    def _autofill_from_ngi(self):
        """Run-seri eşleştirmesine göre yanıt değerlerini NGI veriden doldur"""
        ngi = self.project.ngi_data
        if not ngi:
            QMessageBox.warning(self,"","Önce JSON import edin."); return
        series_map = {s["series_id"]: s for s in ngi.get("series",[])}
        df = self.project.design_matrix
        if df is None: return
        filled = 0
        self._col_blocking = True
        show_cols = [c for c in df.columns if not c.endswith("_coded")]
        for ri, combo in getattr(self,"run_series_combos",{}).items():
            sel = combo.currentText()
            if sel == "(Manuel)" or sel not in series_map: continue
            ser = series_map[sel]
            mean_data = ser.get("mean", {})
            if ri not in self.project.run_results:
                self.project.run_results[ri] = {}
            for resp in self.project.responses:
                if resp in mean_data:
                    val = mean_data[resp].get("mean", np.nan)
                    self.project.run_results[ri][resp] = val
                    # Tabloya yaz
                    col_idx = len(show_cols) + self.project.responses.index(resp)
                    item = self.table.item(ri, col_idx)
                    if not item:
                        item = QTableWidgetItem()
                        self.table.setItem(ri, col_idx, item)
                    item.setText(f"{val:.4f}" if not math.isnan(val) else "")
                    filled += 1
        self._col_blocking = False
        QMessageBox.information(self,"✔",f"{filled} yanıt değeri dolduruldu.")

    def _randomize(self):
        df = self.project.design_matrix
        if df is None: return
        perm = np.random.permutation(len(df))
        self.project.design_matrix = df.iloc[perm].reset_index(drop=True)
        self.project.design_matrix["Run"] = range(1, len(df)+1)
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
            # Yanıt sütunlarını da ekle
            export_df = self.project.build_response_table() or df
            export_df.to_excel(path, index=False)
            QMessageBox.information(self,"✔","Excel kaydedildi.")
        except Exception as e:
            QMessageBox.critical(self,"Hata", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# SEKME 3 — MODEL ANALİZİ
# ═══════════════════════════════════════════════════════════════════════════════
class Tab3_Analysis(QWidget):
    def __init__(self, project: DoEProject, app_ref, parent=None):
        super().__init__(parent)
        self.project = project
        self.app = app_ref
        self._build()

    def _build(self):
        lay = QVBoxLayout(self); lay.setSpacing(10); lay.setContentsMargins(14,14,14,14)

        # Kontrol barı
        bar = QHBoxLayout(); bar.setSpacing(8)
        bar.addWidget(section_label("📊  Model & ANOVA"))
        bar.addStretch()
        self.combo_resp = QComboBox(); self.combo_resp.setFixedWidth(220)
        self.combo_resp.currentTextChanged.connect(self._show_selected)
        bar.addWidget(QLabel("Yanıt:"))
        bar.addWidget(self.combo_resp)
        self.btn_fit = make_btn("▶ Model Fit", "rgba(20,80,20,0.8)", 32)
        self.btn_fit.clicked.connect(self._fit)
        bar.addWidget(self.btn_fit)
        lay.addLayout(bar)

        # Splitter: sol tablo, sağ grafikler
        spl = QSplitter(Qt.Orientation.Horizontal)

        # Sol: ANOVA tablosu + özet
        left = QWidget()
        ll = QVBoxLayout(left); ll.setContentsMargins(0,0,6,0); ll.setSpacing(8)
        ll.addWidget(section_label("ANOVA Tablosu"))
        self.anova_table = QTableWidget()
        self.anova_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        ll.addWidget(self.anova_table)
        ll.addWidget(section_label("Model Özeti"))
        self.txt_summary = QTextEdit()
        self.txt_summary.setReadOnly(True)
        self.txt_summary.setFixedHeight(140)
        self.txt_summary.setStyleSheet(f"""
            QTextEdit {{ background:{BG3}; border:1px solid #2a4060;
                         border-radius:4px; color:{TXT}; font-family:Consolas,monospace; font-size:11px; }}
        """)
        ll.addWidget(self.txt_summary)
        spl.addWidget(left)

        # Sağ: Grafikler
        right = QWidget()
        rl = QVBoxLayout(right); rl.setContentsMargins(6,0,0,0)
        self.fig_analysis = Figure(facecolor=BG2, tight_layout=True)
        self.canvas_analysis = FigureCanvas(self.fig_analysis)
        rl.addWidget(self.canvas_analysis)
        spl.addWidget(right)
        spl.setSizes([420, 560])
        lay.addWidget(spl, 1)

    def refresh(self):
        self.combo_resp.clear()
        for r in self.project.responses:
            self.combo_resp.addItem(RESPONSE_LABELS.get(r, r), r)

    def _fit(self):
        if self.project.design_matrix is None:
            QMessageBox.warning(self,"","Önce tasarım matrisi oluşturun."); return
        # Yanıt verisi var mı kontrol et
        has_data = any(
            self.project.run_results.get(i, {})
            for i in range(len(self.project.design_matrix))
        )
        if not has_data:
            QMessageBox.warning(self,"","Tasarım Matrisi sekmesinden yanıt değerlerini girin."); return
        self.project.fit_models()
        errors = getattr(self.project, 'model_errors', {})
        if errors:
            msg = "\n".join([f"  {k}: {v}" for k,v in errors.items()])
            QMessageBox.warning(self, "Model Uyarıları",
                f"Bazı modeller fit edilemedi:\n{msg}\n\nVeri girişini kontrol edin.")
        if self.project.model_results:
            self._show_selected()
            QMessageBox.information(self, "✔ Model Fit",
                f"{len(self.project.model_results)} model başarıyla fit edildi.")

    def _show_selected(self):
        key = self.combo_resp.currentData()
        if not key: return
        model = self.project.model_results.get(key)
        err = getattr(self.project, 'model_errors', {}).get(key)
        if not model:
            msg = err if err else "Model henüz fit edilmedi. '▶ Model Fit' butonuna basın."
            self.txt_summary.setText(msg)
            self.anova_table.setRowCount(0)
            self.fig_analysis.clear(); self.canvas_analysis.draw(); return

        # ANOVA
        try:
            anova = anova_lm(model, typ=2)
            self._fill_anova(anova)
        except Exception as e:
            print(f"ANOVA error: {e}")

        # Özet
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
                f"BIC      = {model.bic:.2f}"
            )
        except Exception as e:
            self.txt_summary.setText(f"Özet hatası: {e}")

        # Grafikler
        try:
            self.fig_analysis.clear()
            axes = self.fig_analysis.subplots(2, 2)
            self._plot_pareto(axes[0,0], model)
            self._plot_pred_actual(axes[0,1], model)
            self._plot_residuals_normal(axes[1,0], model)
            self._plot_residuals_fitted(axes[1,1], model)
            for ax in axes.flat:
                ax.set_facecolor("#0e1525")
                ax.tick_params(colors=TXT2, labelsize=8)
                for sp in ax.spines.values(): sp.set_color("#2a4060")
            self.fig_analysis.patch.set_facecolor(BG2)
            self.canvas_analysis.draw()
        except Exception as e:
            print(f"Plot error: {e}")

    def _fill_anova(self, anova):
        self.anova_table.setRowCount(len(anova))
        cols = ["df","sum_sq","mean_sq","F","PR(>F)"]
        self.anova_table.setColumnCount(len(cols)+1)
        self.anova_table.setHorizontalHeaderLabels(["Kaynak"]+cols)
        for ri, (idx, row) in enumerate(anova.iterrows()):
            self.anova_table.setItem(ri,0,QTableWidgetItem(str(idx)))
            for ci, col in enumerate(cols):
                val = row.get(col, np.nan)
                if math.isnan(val): txt = "—"
                elif col in ["df"]: txt = str(int(val))
                elif col == "PR(>F)": txt = f"{val:.4f}"
                else: txt = f"{val:.4f}"
                item = QTableWidgetItem(txt)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                # p<0.05 → yeşil
                if col=="PR(>F)" and not math.isnan(val):
                    item.setForeground(QColor("#70e870") if val<0.05 else QColor(TXT))
                self.anova_table.setItem(ri, ci+1, item)
        self.anova_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)

    def _plot_pareto(self, ax, model):
        params = model.params.drop("Intercept", errors="ignore")
        pvals  = model.pvalues.drop("Intercept", errors="ignore")
        t_vals = model.tvalues.drop("Intercept", errors="ignore")
        abs_t  = t_vals.abs().sort_values(ascending=True)
        colors = ["#70e870" if pvals.get(n,1)<0.05 else "#4a6080" for n in abs_t.index]
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
        mn = min(min(actual),min(pred)); mx = max(max(actual),max(pred))
        ax.plot([mn,mx],[mn,mx], color=GOLD, lw=1.2, ls="--")
        ax.set_xlabel("Tahmin", fontsize=8, color=TXT2)
        ax.set_ylabel("Gerçek", fontsize=8, color=TXT2)
        ax.set_title("Tahmin vs Gerçek", fontsize=9, color=TXT)

    def _plot_residuals_normal(self, ax, model):
        resid = model.resid
        (osm, osr), (slope, intercept, r) = stats.probplot(resid)
        ax.plot(osm, osr, "o", color="#ED7D31", ms=5, alpha=0.8)
        ax.plot(osm, slope*np.array(osm)+intercept, color=GOLD, lw=1.2, ls="--")
        ax.set_title("Normal Olasılık — Artıklar", fontsize=9, color=TXT)
        ax.set_xlabel("Teorik kantil", fontsize=8, color=TXT2)
        ax.set_ylabel("Artık", fontsize=8, color=TXT2)

    def _plot_residuals_fitted(self, ax, model):
        ax.scatter(model.fittedvalues, model.resid, color="#70AD47", s=30, alpha=0.8)
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
        self._build()

    def _build(self):
        lay = QVBoxLayout(self); lay.setSpacing(10); lay.setContentsMargins(14,14,14,14)

        # Kontrol
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

        # Sabit faktörler
        self.fixed_card = card_frame()
        fl = QHBoxLayout(self.fixed_card); fl.setContentsMargins(10,8,10,8); fl.setSpacing(12)
        fl.addWidget(QLabel("Sabit faktörler:"))
        self.fixed_widgets = {}
        fl.addStretch()
        lay.addWidget(self.fixed_card)

        # Grafik alanı
        self.fig_surf = Figure(facecolor=BG2)
        self.canvas_surf = FigureCanvas(self.fig_surf)
        lay.addWidget(self.canvas_surf, 1)

    def refresh(self):
        self.combo_resp.clear()
        self.combo_x.clear(); self.combo_y.clear()
        for r in self.project.responses:
            self.combo_resp.addItem(RESPONSE_LABELS.get(r,r), r)
        for f in self.project.factors:
            if f["type"] == "continuous":
                self.combo_x.addItem(f["name"])
                self.combo_y.addItem(f["name"])
        if self.combo_y.count() > 1:
            self.combo_y.setCurrentIndex(1)
        self._update_fixed()

    def _update_fixed(self):
        # Sabit faktör spinbox'larını güncelle
        for w in self.fixed_widgets.values():
            w.setParent(None)
        self.fixed_widgets.clear()
        x_name = self.combo_x.currentText()
        y_name = self.combo_y.currentText()
        fl = self.fixed_card.layout()
        for item in reversed(range(2, fl.count()-1)):
            w = fl.itemAt(item).widget()
            if w: w.setParent(None)
        for f in self.project.factors:
            if f["name"] in [x_name, y_name]: continue
            if f["type"] != "continuous": continue
            mid = (f["low"] + f["high"]) / 2
            grp = QWidget(); grp.setStyleSheet("background:transparent;")
            gl = QHBoxLayout(grp); gl.setContentsMargins(0,0,0,0); gl.setSpacing(4)
            gl.addWidget(QLabel(f["name"]+":"))
            sp = QDoubleSpinBox()
            sp.setRange(f["low"], f["high"]); sp.setValue(mid)
            sp.setDecimals(3); sp.setFixedWidth(90)
            gl.addWidget(sp)
            self.fixed_widgets[f["name"]] = sp
            fl.insertWidget(fl.count()-1, grp)

    def _plot(self):
        resp_key = self.combo_resp.currentData()
        model = self.project.model_results.get(resp_key)
        if not model:
            QMessageBox.warning(self,"","Önce model fit edin (Analiz sekmesi)."); return

        x_name = self.combo_x.currentText()
        y_name = self.combo_y.currentText()
        if x_name == y_name:
            QMessageBox.warning(self,"","X ve Y aynı faktör olamaz."); return

        x_f = next((f for f in self.project.factors if f["name"]==x_name), None)
        y_f = next((f for f in self.project.factors if f["name"]==y_name), None)
        if not x_f or not y_f: return

        n = 40
        xs = np.linspace(x_f["low"], x_f["high"], n)
        ys = np.linspace(y_f["low"], y_f["high"], n)
        XX, YY = np.meshgrid(xs, ys)

        # Tahmin için DataFrame oluştur
        safe = {f["name"].replace(" ","_").replace("-","_"): f["name"] for f in self.project.factors}
        # predict_at kullanarak tutarlı tahmin yap
        ZZ_flat = []
        for xi, yi in zip(XX.ravel(), YY.ravel()):
            actual_vals = []
            for f in self.project.factors:
                if f["name"] == x_name:
                    actual_vals.append(xi)
                elif f["name"] == y_name:
                    actual_vals.append(yi)
                elif f["name"] in self.fixed_widgets:
                    actual_vals.append(self.fixed_widgets[f["name"]].value())
                else:
                    actual_vals.append((f["low"]+f["high"])/2)
            pred, _ = self.project.predict_at(resp_key, actual_vals)
            ZZ_flat.append(pred if pred is not None else float('nan'))

        try:
            ZZ = np.array(ZZ_flat).reshape(n, n)
        except Exception as e:
            QMessageBox.critical(self,"Tahmin Hatası",str(e)); return

        self.fig_surf.clear()
        ax3 = self.fig_surf.add_subplot(121, projection="3d")
        ax2 = self.fig_surf.add_subplot(122)
        self.fig_surf.patch.set_facecolor(BG2)

        # 3D yüzey
        ax3.plot_surface(XX, YY, ZZ, cmap="coolwarm", alpha=0.85, edgecolor="none")
        ax3.set_xlabel(x_name, fontsize=8, color=TXT2)
        ax3.set_ylabel(y_name, fontsize=8, color=TXT2)
        ax3.set_zlabel(RESPONSE_LABELS.get(resp_key,resp_key), fontsize=7, color=TXT2)
        ax3.set_title("Response Surface", fontsize=9, color=TXT)
        ax3.set_facecolor("#0e1525")
        ax3.tick_params(colors=TXT2, labelsize=7)

        # 2D kontur
        cp = ax2.contourf(XX, YY, ZZ, levels=15, cmap="coolwarm")
        self.fig_surf.colorbar(cp, ax=ax2, shrink=0.8)
        ax2.contour(XX, YY, ZZ, levels=8, colors="white", alpha=0.3, linewidths=0.5)
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
        self._build()

    def _build(self):
        lay = QVBoxLayout(self); lay.setSpacing(10); lay.setContentsMargins(14,14,14,14)
        lay.addWidget(section_label("🎯  Desirability Optimizasyonu"))

        # Hedef tanımlama kartı
        tgt_card = card_frame()
        tl = QVBoxLayout(tgt_card); tl.setContentsMargins(12,10,12,10); tl.setSpacing(6)
        tl.addWidget(section_label("Yanıt Hedefleri"))
        self.tgt_scroll = QScrollArea()
        self.tgt_scroll.setWidgetResizable(True); self.tgt_scroll.setFixedHeight(160)
        self.tgt_scroll.setStyleSheet("border:none; background:transparent;")
        self.tgt_container = QWidget(); self.tgt_container.setStyleSheet("background:transparent;")
        self.tgt_layout = QVBoxLayout(self.tgt_container)
        self.tgt_layout.setSpacing(4); self.tgt_layout.setContentsMargins(0,0,0,0)
        self.tgt_layout.addStretch()
        self.tgt_scroll.setWidget(self.tgt_container)
        tl.addWidget(self.tgt_scroll)
        self.tgt_rows = {}
        lay.addWidget(tgt_card)

        # Optimize et butonu
        bar = QHBoxLayout(); bar.addStretch()
        self.btn_opt = make_btn("🚀 Optimize Et", "rgba(20,80,20,0.8)", 36)
        self.btn_opt.clicked.connect(self._optimize)
        bar.addWidget(self.btn_opt)
        lay.addLayout(bar)

        # Sonuçlar
        res_card = card_frame()
        rl = QVBoxLayout(res_card); rl.setContentsMargins(12,10,12,10); rl.setSpacing(8)
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
        for w in self.tgt_rows.values(): w.setParent(None)
        self.tgt_rows.clear()
        for resp in self.project.responses:
            row = QWidget(); row.setStyleSheet("background:transparent;")
            rl = QHBoxLayout(row); rl.setContentsMargins(0,0,0,0); rl.setSpacing(8)
            lbl = QLabel(RESPONSE_LABELS.get(resp,resp))
            lbl.setFixedWidth(160)
            combo = QComboBox()
            combo.addItems(["Minimize Et","Maximize Et","Hedefe Ulaş"])
            combo.setFixedWidth(140)
            target_sp = QDoubleSpinBox()
            target_sp.setRange(-1e6,1e6); target_sp.setValue(0)
            target_sp.setFixedWidth(100)
            target_sp.setEnabled(False)
            combo.currentTextChanged.connect(
                lambda t, sp=target_sp: sp.setEnabled(t=="Hedefe Ulaş"))
            rl.addWidget(lbl); rl.addWidget(combo); rl.addWidget(target_sp); rl.addStretch()
            self.tgt_rows[resp] = {"combo":combo, "target":target_sp, "widget":row}
            self.tgt_layout.insertWidget(self.tgt_layout.count()-1, row)

    def _optimize(self):
        if not self.project.model_results:
            QMessageBox.warning(self,"","Önce Analiz sekmesinden 'Model Fit' yapın."); return

        from scipy.optimize import differential_evolution

        # Sadece sürekli faktörler optimize edilir
        all_factors = self.project.factors
        cont_idx = [i for i,f in enumerate(all_factors) if f["type"]=="continuous"]
        if not cont_idx:
            QMessageBox.warning(self,"","Optimize edilecek sürekli faktör yok."); return

        cont_factors = [all_factors[i] for i in cont_idx]
        bounds = [(f["low"], f["high"]) for f in cont_factors]

        # Yanıt aralıklarını önceden hesapla (desirability için)
        resp_ranges = {}
        df2 = self.project.build_response_table()
        for resp in self.project.responses:
            if df2 is not None and resp in df2.columns:
                col_vals = df2[resp].dropna()
                if len(col_vals) >= 2:
                    resp_ranges[resp] = (float(col_vals.min()), float(col_vals.max()))

        def desirability(x_cont):
            # Tüm faktörler için actual değer listesi oluştur
            actual_vals = []
            cont_ptr = 0
            for i, f in enumerate(all_factors):
                if i in cont_idx:
                    actual_vals.append(x_cont[cont_ptr])
                    cont_ptr += 1
                else:
                    actual_vals.append((f["low"] + f["high"]) / 2)

            d_vals = []
            for resp, info in self.tgt_rows.items():
                if resp not in self.project.model_results: continue
                pred, _ = self.project.predict_at(resp, actual_vals)
                if pred is None: continue
                goal = info["combo"].currentText()
                lo, hi = resp_ranges.get(resp, (pred-1, pred+1))
                span = hi - lo if hi != lo else 1.0
                if goal == "Minimize Et":
                    d = max(0.0, min(1.0, (hi - pred) / span))
                elif goal == "Maximize Et":
                    d = max(0.0, min(1.0, (pred - lo) / span))
                else:  # Hedefe Ulaş
                    t = info["target"].value()
                    d = max(0.0, 1.0 - abs(pred - t) / span)
                d_vals.append(d)
            if not d_vals: return 1.0
            return -(np.prod(d_vals) ** (1.0 / len(d_vals)))

        try:
            result = differential_evolution(
                desirability, bounds, seed=42,
                maxiter=800, tol=1e-7, polish=True,
                popsize=15, mutation=(0.5, 1.5), recombination=0.7
            )
            x_opt = result.x
            d_opt = -result.fun
        except Exception as e:
            QMessageBox.critical(self,"Optimizasyon Hatası", str(e)); return

        # Tüm faktörler için actual değer listesi
        actual_opt = []
        cont_ptr = 0
        for i, f in enumerate(all_factors):
            if i in cont_idx:
                actual_opt.append(x_opt[cont_ptr]); cont_ptr += 1
            else:
                actual_opt.append((f["low"] + f["high"]) / 2)

        # Sonuç metni
        lines = [
            "─" * 52,
            "   OPTİMUM FORMÜLASYON ÖNERİSİ",
            "─" * 52, "",
            "  FAKTÖR DEĞERLERİ:"
        ]
        for i, f in enumerate(all_factors):
            unit = f"  {f['unit']}" if f.get("unit") else ""
            lines.append(f"    {f['name']:28s} = {actual_opt[i]:.4f}{unit}")

        lines += ["", "  TAHMİN EDİLEN YANIT DEĞERLERİ:"]
        for resp in self.project.responses:
            lbl = RESPONSE_LABELS.get(resp, resp)
            if resp not in self.project.model_results:
                lines.append(f"    {lbl:32s} = (model yok)")
                continue
            pred, ci = self.project.predict_at(resp, actual_opt)
            if pred is None:
                lines.append(f"    {lbl:32s} = (tahmin hatası)")
            else:
                ci_str = f"  [%95 PI: {ci[0]:.4f} – {ci[1]:.4f}]" if ci else ""
                lines.append(f"    {lbl:32s} = {pred:.4f}{ci_str}")

        lines += [
            "",
            f"  Genel Desirability  = {d_opt:.4f}  "
            f"({'Mükemmel' if d_opt>0.8 else 'İyi' if d_opt>0.6 else 'Kabul Edilebilir' if d_opt>0.4 else 'Zayıf'})",
            "─" * 52
        ]
        self.txt_opt_result.setText("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════════════
# ANA UYGULAMA
# ═══════════════════════════════════════════════════════════════════════════════
class DoEApp(QMainWindow):
    def __init__(self, json_path=None):
        super().__init__()
        self.project = DoEProject()
        self.setWindowTitle("NGI DoE Analyzer")
        self.setMinimumSize(1200, 760)
        # Pencere ikonu
        ico_path = resource_path("doe_icon.ico")
        if os.path.exists(ico_path):
            self.setWindowIcon(QIcon(ico_path))
        self._build_ui()
        self.setStyleSheet(STYLE)
        if json_path and os.path.exists(json_path):
            QTimer.singleShot(300, lambda: self.tab1.load_json(json_path))

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        ml = QVBoxLayout(central); ml.setContentsMargins(0,0,0,0); ml.setSpacing(0)

        # Başlık
        hdr = QWidget()
        hdr.setFixedHeight(52)
        hdr.setStyleSheet(f"background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
                          f"stop:0 #0a1628, stop:1 #0e1f3a);")
        hl = QHBoxLayout(hdr); hl.setContentsMargins(18,0,18,0); hl.setSpacing(10)
        icon_lbl = QLabel("🔬")
        icon_lbl.setStyleSheet("font-size:22px; background:transparent;")
        hl.addWidget(icon_lbl)
        t1 = QLabel("NGI DoE Analyzer")
        t1.setStyleSheet(f"color:white; font-size:16px; font-weight:bold; background:transparent;")
        hl.addWidget(t1)
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet(f"color:#2a4060; background:#2a4060;"); sep.setFixedWidth(1)
        hl.addWidget(sep)
        t2 = QLabel("Design of Experiments — NGI-CITDAS Entegre")
        t2.setStyleSheet(f"color:{TXT2}; font-size:11px; background:transparent;")
        hl.addWidget(t2)
        hl.addStretch()
        self.lbl_status = QLabel("Hazır.")
        self.lbl_status.setStyleSheet(f"color:{TXT2}; font-size:11px; background:transparent;")
        hl.addWidget(self.lbl_status)
        ml.addWidget(hdr)

        # Sekmeler
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        ml.addWidget(self.tabs, 1)

        self.tab1 = Tab1_Factors(self.project, self)
        self.tab2 = Tab2_Design(self.project, self)
        self.tab3 = Tab3_Analysis(self.project, self)
        self.tab4 = Tab4_Surface(self.project, self)
        self.tab5 = Tab5_Optimization(self.project, self)

        self.tabs.addTab(self.tab1, "1 · Faktörler & Tasarım")
        self.tabs.addTab(self.tab2, "2 · Tasarım Matrisi")
        self.tabs.addTab(self.tab3, "3 · Model & ANOVA")
        self.tabs.addTab(self.tab4, "4 · Response Surface")
        self.tabs.addTab(self.tab5, "5 · Optimizasyon")

    def refresh_all_tabs(self):
        self.tab2.refresh()
        self.tab3.refresh()
        self.tab4.refresh()
        self.tab5.refresh()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("NGI DoE Analyzer")
    json_arg = sys.argv[1] if len(sys.argv) > 1 else None
    win = DoEApp(json_arg)
    win.show()
    sys.exit(app.exec())
