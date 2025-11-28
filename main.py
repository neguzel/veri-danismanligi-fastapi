import os
import io
import json
import textwrap
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, Request, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    Float,
    Boolean,
    ForeignKey,
    Text,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session as OrmSession

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
client: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
CHART_DIR = os.path.join(STATIC_DIR, "charts")
REPORT_DIR = os.path.join(STATIC_DIR, "reports")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# PDF için Türkçe karakter desteği olan font kaydı
try:
    pdfmetrics.registerFont(TTFont("ArialTR", "C:/Windows/Fonts/arial.ttf"))
    PDF_FONT = "ArialTR"
except Exception:
    PDF_FONT = "Helvetica"

# OpenAI client (API key yoksa None)
client: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# -------------------------------------------------------------------
# Veritabanı
# -------------------------------------------------------------------

DATABASE_URL = "sqlite:///" + os.path.join(BASE_DIR, "veridanismanligi.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    # Temel giriş bilgileri
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)  # Demo: düz şifre, prod için hash önerilir
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Profil / iletişim bilgileri
    full_name = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    company = Column(String, nullable=True)
    sector = Column(String, nullable=True)

    uploads = relationship("Upload", back_populates="user")


class Upload(Base):
    __tablename__ = "uploads"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    file_name = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    company = Column(String, nullable=True)

    # İletişim bilgileri (login yok, sadece upload anında alınır)
    contact_name = Column(String, nullable=True)
    contact_phone = Column(String, nullable=True)
    contact_email = Column(String, nullable=True)
    contact_sector = Column(String, nullable=True)

    row_count = Column(Integer, default=0)
    col_count = Column(Integer, default=0)
    total_cells = Column(Integer, default=0)
    total_missing = Column(Integer, default=0)
    quality_score = Column(Float, default=0.0)
    top_missing_col = Column(String, nullable=True)
    top_var_col = Column(String, nullable=True)
    domain_insights = Column(Text, nullable=True)

    ai_summary = Column(Text, nullable=True)
    ai_risks = Column(Text, nullable=True)
    ai_features = Column(Text, nullable=True)
    ai_models = Column(Text, nullable=True)
    ai_recommendations = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="uploads")


Base.metadata.create_all(bind=engine)

# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------

app = FastAPI(title="Veri Danışmanlığı – Akıllı Veri Analiz Paneli")
app.add_middleware(SessionMiddleware, secret_key="CHANGE_THIS_SECRET")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

ANALYSIS_CACHE: Dict[int, Dict[str, Any]] = {}


# -------------------------------------------------------------------
# DB dependency & yardımcılar
# -------------------------------------------------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def current_user(request: Request, db: OrmSession) -> Optional[User]:
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    return db.query(User).filter(User.id == user_id).first()


# -------------------------------------------------------------------
# AI analizi (OpenAI SDK + JSON)
# -------------------------------------------------------------------

AI_SYSTEM_PROMPT = """
Sen üst düzey bir veri bilimi danışmanısın.
Tüm analizleri profesyonel, sade ve yöneticilere uygun Türkçe ile yaparsın.

Kullanıcıdan veri setine ait özet bilgiler alacaksın.
Bu bilgiler: satır/kolon sayıları, eksik veri oranı, varyans, alan tipleri, sektör vb. olabilir.

Sana yüklediğim veri setlerinde ilgili verileri analiz et. Analizini yaparken seçilen sektör dinamiklerine göre yorumlar yap. 
(sağladığım datanın kalitesinden ziyade veriyi anlamlandır.) Bana vereceğin bilgiler ışığında ben firmalara çözüm önerileri sunmak istiyorum. 
“Uygulanabilir Model Önerileri” kısmında firma verilerin analizi sonucu hangi önerini yaparsa karlılık ve verimlilik arttırır bunu dikkate alacak.” 
“İş / Veri Geliştirme Önerileri” kısmında da verdiğin bilgiler ışığında firma kendisine yol haritası çizecek.

⛔ Kurallar:
- ÇIKTI HER ZAMAN GEÇERLİ BİR JSON NESNESİ OLACAK.
- Kod bloğu, markdown, ```json veya başka bir format KULLANMA.
- JSON dışında TEK BİR KARAKTER BİLE yazma.
- Değerler TÜRKÇE olacak, key isimleri İNGİLİZCE kalacak.

🎯 Üreteceğin JSON şeması:

{
  "summary": "<genel kısa özet>",
  "risks": ["<risk 1>", "<risk 2>", ...],
  "features": ["<öneri 1>", "<öneri 2>", ...],
  "ml_models": ["<model önerisi>", ...],
  "recommendations": ["<aksiyon önerisi>", ...]
}

Sektör bilgisi varsa (enerji, gıda, çelik, plastik, otomotiv, tekstil, sağlık, finans, lojistik, kimya vb.)
yorumları sektöre uygunlaştır.
"""


def _join_list_or_str(value: Any) -> str:
    """LLM'den gelen liste/string değerleri her zaman stringe çevirir."""
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(f"- {str(item)}" for item in value if str(item).strip())
    return str(value)


def ai_analyze_dataframe(df: pd.DataFrame, sector: Optional[str] = None) -> Dict[str, str]:
    """
    Veri seti için sektör bağımsız, yapısal AI analizi.
    Dönen değerler: summary/risks/features/ml_models/recommendations -> hepsi string.
    """
    rows, cols = df.shape
    missing_total = int(df.isna().sum().sum())
    total_cells = max(rows * cols, 1)
    missing_ratio = round((missing_total / total_cells) * 100, 2)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # En yüksek varyanslı ilk 5 kolon
    high_var: List[str] = []
    if numeric_cols:
        var_series = df[numeric_cols].var(numeric_only=True).sort_values(ascending=False)
        for c, v in var_series.head(5).items():
            high_var.append(f"{c} (var={round(v, 2)})")

    summary_text = f"""
Dosya Özeti:
- Sektör: {sector or 'belirtilmemiş'}
- Satır sayısı: {rows}
- Kolon sayısı: {cols}
- Toplam eksik hücre: {missing_total} (%{missing_ratio})
- Sayısal kolonlar: {', '.join(numeric_cols) if numeric_cols else '-'}
- Kategorik kolonlar: {', '.join(cat_cols) if cat_cols else '-'}
- En yüksek varyansa sahip alanlar: {', '.join(high_var) if high_var else '-'}
""".strip()

    # API anahtarı yoksa demo cevap
    if not client:
        risks_list = [
            "Gerçek zamanlı AI analizi devre dışı (API anahtarı tanımsız).",
            "Eksik veri, aykırı değerler ve iş kuralları manuel olarak kontrol edilmelidir.",
        ]
        features_list = [
            "Sayısal değişkenler için normalizasyon / standardizasyon.",
            "Kategori alanları için etiket kodlama (one-hot veya target encoding).",
        ]
        models_list = [
            "Temel regresyon / sınıflandırma modelleri (Linear Regression, Logistic Regression).",
            "Ağaç tabanlı modeller (Random Forest, XGBoost, LightGBM).",
        ]
        recs_list = [
            "OpenAI API anahtarı eklendiğinde tam AI raporları otomatik üretilecektir.",
            "Pilot proje için küçük bir veri alt kümesi ile ilk modelleme denemeleri yapılabilir.",
        ]
        return {
            "summary": "Demo mod: OpenAI API anahtarı tanımlı olmadığı için yerel özet gösteriliyor.",
            "risks": _join_list_or_str(risks_list),
            "features": _join_list_or_str(features_list),
            "ml_models": _join_list_or_str(models_list),
            "recommendations": _join_list_or_str(recs_list),
        }

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            temperature=0.1,
            messages=[
                {"role": "system", "content": AI_SYSTEM_PROMPT},
                {"role": "user", "content": summary_text},
            ],
        )

        content = (response.choices[0].message.content or "").strip()
        data = json.loads(content)

        return {
            "summary": _join_list_or_str(data.get("summary")),
            "risks": _join_list_or_str(data.get("risks")),
            "features": _join_list_or_str(data.get("features")),
            "ml_models": _join_list_or_str(data.get("ml_models")),
            "recommendations": _join_list_or_str(data.get("recommendations")),
        }

    except Exception as e:
        err_type = type(e).__name__
        err_msg = str(e)
        return {
            "summary": f"AI çalıştırılamadı ({err_type}).",
            "risks": f"OpenAI hatası: {err_msg}",
            "features": "-",
            "ml_models": "-",
            "recommendations": "-",
        }


# -------------------------------------------------------------------
# AI destekli grafik üretimi
# -------------------------------------------------------------------

def suggest_charts_with_ai(df: pd.DataFrame, max_charts: int = 6) -> List[Dict[str, Any]]:
    """
    DataFrame yapısına bakarak OpenAI'den grafik önerileri ister.
    Tipler: "hist", "bar", "line", "pie", "box", "heatmap"
    """
    # API key yoksa grafik önerme
    if not client:
        return []

    # Şema özeti
    schema_info: List[Dict[str, Any]] = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        nunique = int(df[col].nunique())
        schema_info.append(
            {
                "name": col,
                "dtype": dtype,
                "nunique": nunique,
            }
        )

    # Sayısal özet
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_summary: Dict[str, Any] = {}
    if numeric_cols:
        desc = df[numeric_cols].describe().T
        for col in numeric_cols:
            if col in desc.index:
                row = desc.loc[col]
                numeric_summary[col] = {
                    "mean": float(row.get("mean", 0.0)),
                    "std": float(row.get("std", 0.0)),
                    "min": float(row.get("min", 0.0)),
                    "max": float(row.get("max", 0.0)),
                }

    system_prompt = """
Sen bir veri görselleştirme asistanısın. Görevin:
- Verilen tablo şemasına (kolon adları, veri tipleri, özet istatistikler) bakarak
- En fazla N adet (max_charts) anlamlı grafik önerisi yapmak.
- Sadece şu tipleri kullan: "hist", "bar", "line", "pie", "box", "heatmap".
- Çıktıyı KESİNLİKLE saf JSON liste olarak ver. Başına/sonuna açıklama ekleme.

Her grafik için zorunlu alanlar:
- "id": Benzersiz bir id (ör: "chart_1")
- "type": "hist" | "bar" | "line" | "pie" | "box" | "heatmap"
- "columns": Kullandığın kolon(lar) listesi
- "title": Kısa ve anlaşılır Türkçe başlık
- "description": 1-2 cümlelik Türkçe açıklama
"""

    user_content = {
        "schema": schema_info,
        "numeric_summary": numeric_summary,
        "max_charts": max_charts,
    }

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)},
        ],
    )

    raw = resp.choices[0].message.content or ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if isinstance(data, dict) and "charts" in data:
        charts = data["charts"]
    else:
        charts = data

    if not isinstance(charts, list):
        return []

    valid_types = {"hist", "bar", "line", "pie", "box", "heatmap"}
    cleaned: List[Dict[str, Any]] = []
    for i, ch in enumerate(charts, start=1):
        ctype = str(ch.get("type", "")).lower()
        cols = ch.get("columns") or []
        if ctype not in valid_types or not cols:
            continue
        cleaned.append(
            {
                "id": ch.get("id") or f"chart_{i}",
                "type": ctype,
                "columns": cols,
                "title": ch.get("title") or f"Grafik {i}",
                "description": ch.get("description", ""),
            }
        )

    return cleaned


def render_chart_from_spec(
    df: pd.DataFrame,
    upload_id: int,
    spec: Dict[str, Any],
) -> Optional[str]:
    """
    AI'den gelen grafik tanımını kullanarak PNG üretir.
    Dönüş: /static/charts/... şeklinde URL (veya None).
    """
    chart_type = spec["type"]
    cols = spec["columns"]
    title = spec.get("title") or "Grafik"
    chart_id = spec.get("id") or "chart"

    plt.figure()

    try:
        if chart_type == "hist":
            col = cols[0]
            df[col].dropna().hist(bins=30)
            plt.xlabel(col)
            plt.ylabel("Frekans")

        elif chart_type == "line":
            if len(cols) == 1:
                col = cols[0]
                df[col].dropna().reset_index(drop=True).plot()
                plt.xlabel("Index")
                plt.ylabel(col)
            else:
                x, y = cols[0], cols[1]
                df.dropna(subset=[x, y]).plot(x=x, y=y)
                plt.xlabel(x)
                plt.ylabel(y)

        elif chart_type == "bar":
            if len(cols) >= 2:
                cat_col, val_col = cols[0], cols[1]
                tmp = df[[cat_col, val_col]].dropna()
                if not tmp.empty:
                    agg = (
                        tmp.groupby(cat_col)[val_col]
                        .mean()
                        .sort_values(ascending=False)
                        .head(20)
                    )
                    agg.plot(kind="bar")
                    plt.xlabel(cat_col)
                    plt.ylabel(f"{val_col} (ortalama)")
            else:
                col = cols[0]
                df[col].dropna().plot(kind="hist", bins=20)
                plt.xlabel(col)
                plt.ylabel("Frekans")

        elif chart_type == "pie":
            col = cols[0]
            s = df[col].dropna().value_counts()
            if s.empty:
                plt.text(0.5, 0.5, "Veri yok", ha="center", va="center")
            else:
                s = s.head(8)
                if len(s) > 6:
                    top = s[:5]
                    other = s[5:].sum()
                    s = top.append(pd.Series({"Diğer": other}))
                s.plot(kind="pie", autopct="%1.1f%%")
                plt.ylabel("")

        elif chart_type == "box":
            selected = [c for c in cols if c in df.columns]
            if selected:
                df[selected].dropna().plot(kind="box")
                plt.xticks(rotation=45)

        elif chart_type == "heatmap":
            numeric = df.select_dtypes(include="number")
            if numeric.shape[1] >= 2:
                corr = numeric.corr()
                plt.imshow(corr, interpolation="nearest")
                plt.colorbar()
                plt.xticks(
                    range(len(corr.columns)),
                    corr.columns,
                    rotation=45,
                    ha="right",
                )
                plt.yticks(range(len(corr.columns)), corr.columns)
            else:
                if numeric.shape[1] == 1:
                    col = numeric.columns[0]
                    numeric[col].dropna().hist(bins=30)
                    plt.xlabel(col)
                    plt.ylabel("Frekans")
                else:
                    plt.text(
                        0.5,
                        0.5,
                        "Korelasyon için yeterli sayısal kolon yok",
                        ha="center",
                        va="center",
                    )

        plt.title(title)
        plt.tight_layout()

        filename = f"{upload_id}_{chart_id}_{chart_type}.png"
        filepath = os.path.join(CHART_DIR, filename)
        plt.savefig(filepath)
        plt.close()

        return f"/static/charts/{filename}"

    except Exception:
        plt.close()
        return None


def generate_charts(df: pd.DataFrame, upload_id: int) -> Dict[str, Any]:
    """
    AI destekli grafik üretimi.
    - OpenAI'den farklı tiplerde grafik şablonları istenir.
    - Gelen şablonlara göre grafikler çizilir.
    - Hiç grafik üretilemezse eski davranışa (histogram + trend) düşer.
    """
    chart_cards: List[Dict[str, Any]] = []
    hist_paths: List[str] = []
    trend_url: Optional[str] = None

    # 1) AI'den grafik önerilerini al
    try:
        specs = suggest_charts_with_ai(df, max_charts=6)
    except Exception:
        specs = []

    # 2) AI önerilerine göre grafik çiz
    if specs:
        for spec in specs:
            url = render_chart_from_spec(df, upload_id, spec)
            if not url:
                continue
            card = {
                "title": spec.get("title") or "Grafik",
                "url": url,
                "description": spec.get("description", ""),
                "type": spec.get("type"),
            }
            chart_cards.append(card)
            hist_paths.append(url)

    # 3) Hiç grafik oluşmadıysa fallback
    if not chart_cards:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        for col in numeric_cols:
            plt.figure()
            df[col].dropna().hist(bins=30)
            plt.title(f"{col} - Dağılım")
            plt.xlabel(col)
            plt.ylabel("Frekans")
            plt.tight_layout()

            filename = f"{upload_id}_hist_{col}.png"
            filepath = os.path.join(CHART_DIR, filename)
            plt.savefig(filepath)
            plt.close()

            url = f"/static/charts/{filename}"
            hist_paths.append(url)
            chart_cards.append({"title": f"{col} – Dağılım", "url": url})

        if numeric_cols:
            col = numeric_cols[0]
            plt.figure()
            df[col].reset_index(drop=True).plot()
            plt.title(f"{col} - Trend")
            plt.xlabel("Index")
            plt.ylabel(col)
            plt.tight_layout()

            filename = f"{upload_id}_trend_{col}.png"
            filepath = os.path.join(CHART_DIR, filename)
            plt.savefig(filepath)
            plt.close()

            trend_url = f"/static/charts/{filename}"
            chart_cards.append({"title": f"{col} – Trend", "url": trend_url})

    return {
        "charts": chart_cards,
        "histograms": hist_paths,
        "trend": trend_url,
    }


def build_chart_cards(charts: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Template'te kullanılacak kart yapısını üretir.
    - Yeni "charts" yapısını direkt kullanır.
    - Yoksa eski "histograms + trend" mantığına döner.
    """
    cards: List[Dict[str, Any]] = []
    if not charts:
        return cards

    ai_cards = charts.get("charts")
    if ai_cards:
        for c in ai_cards:
            cards.append(
                {
                    "title": c.get("title", "Grafik"),
                    "url": c.get("url"),
                    "description": c.get("description", ""),
                    "type": c.get("type"),
                }
            )
        return cards

    histos = charts.get("histograms") or []
    for url in histos:
        base = os.path.basename(url)
        name = os.path.splitext(base)[0]
        parts = name.split("_")
        col = parts[-1] if len(parts) > 2 else ""
        title = f"{col} – Dağılım" if col else "Dağılım Grafiği"
        cards.append({"title": title, "url": url})

    trend_url = charts.get("trend")
    if trend_url:
        base = os.path.basename(trend_url)
        name = os.path.splitext(base)[0]
        parts = name.split("_")
        col = parts[-1] if len(parts) > 2 else ""
        title = f"{col} – Trend" if col else "Trend Grafiği"
        cards.append({"title": title, "url": trend_url})

    return cards


# -------------------------------------------------------------------
# PDF üretimi
# -------------------------------------------------------------------

def generate_pdf_report(
    output_path: str,
    summary: str,
    risks: str,
    features: str,
    models: str,
    recommendations: str,
    chart_files: Optional[List[str]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Daha düzenli, UX/UI odaklı PDF rapor üretir.
    """
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    margin = 2 * cm

    def new_page_header(title: str) -> float:
        c.showPage()
        c.setFont(PDF_FONT, 16)
        y_ = height - margin
        c.drawString(margin, y_, title)
        return y_ - 1.2 * cm

    def draw_section_title(txt: str, y_: float) -> float:
        if y_ < 3 * cm:
            y_ = new_page_header("Veri Analiz Raporu")
        c.setFont(PDF_FONT, 12)
        c.drawString(margin, y_, txt)
        c.setLineWidth(0.3)
        c.line(margin, y_ - 0.15 * cm, width - margin, y_ - 0.15 * cm)
        return y_ - 0.6 * cm

    def draw_paragraph(text: str, y_: float, font_size: int = 10) -> float:
        if not text:
            return y_
        c.setFont(PDF_FONT, font_size)
        max_chars = 110
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                y_ -= 0.4 * cm
                continue
            wrapped = textwrap.wrap(line, max_chars) or [line]
            for wline in wrapped:
                if y_ < 2.5 * cm:
                    y_ = new_page_header("Veri Analiz Raporu (devam)")
                c.drawString(margin, y_, wline)
                y_ -= 0.45 * cm
        y_ -= 0.3 * cm
        return y_

    # Başlık + tarih
    c.setFont(PDF_FONT, 18)
    y = height - margin
    c.drawString(margin, y, "Veri Analiz Raporu")

    c.setFont(PDF_FONT, 9)
    created_text = f"Oluşturulma Tarihi: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} (UTC)"
    c.drawRightString(width - margin, y, created_text)

    y -= 1.2 * cm

    # Müşteri / firma kutusu
    if meta:
        c.setFont(PDF_FONT, 11)
        box_top = y
        box_bottom = y - 3.2 * cm
        if box_bottom < 2 * cm:
            box_bottom = 2 * cm
        c.setLineWidth(0.6)
        c.rect(margin, box_bottom, width - 2 * margin, box_top - box_bottom, stroke=1, fill=0)

        y_line = box_top - 0.8 * cm

        def meta_line(label: str, key: str):
            nonlocal y_line
            val = (meta.get(key) or "").strip()
            if not val:
                return
            c.drawString(margin + 0.4 * cm, y_line, f"{label}: {val}")
            y_line -= 0.55 * cm

        meta_line("Firma", "company")
        meta_line("Ad Soyad", "contact_name")
        meta_line("E-posta", "contact_email")
        meta_line("Telefon", "contact_phone")
        meta_line("Sektör", "contact_sector")

        y = box_bottom - 0.8 * cm
    else:
        y -= 0.4 * cm

    sections = [
        ("AI Özet", summary),
        ("Riskler", risks),
        ("Feature Engineering Önerileri", features),
        ("Uygun ML Modelleri", models),
        ("Aksiyon Önerileri", recommendations),
    ]

    for title, text in sections:
        if text and text.strip():
            y = draw_section_title(title, y)
            y = draw_paragraph(text, y)

    # Grafikler
    if chart_files:
        y = new_page_header("Grafikler")
        c.setFont(PDF_FONT, 10)

        for idx, chart_path in enumerate(chart_files, start=1):
            if y < 8 * cm:
                y = new_page_header("Grafikler")

            chart_name = os.path.basename(chart_path)
            c.setFont(PDF_FONT, 11)
            c.drawString(margin, y, f"Grafik {idx}: {chart_name}")
            y -= 0.6 * cm

            try:
                img_height = 9 * cm
                img_width = width - 2 * margin
                c.drawImage(
                    chart_path,
                    margin,
                    y - img_height,
                    width=img_width,
                    height=img_height,
                    preserveAspectRatio=True,
                    anchor="n",
                )
                y -= img_height + 1 * cm
            except Exception:
                c.setFont(PDF_FONT, 9)
                c.drawString(margin, y, "(Grafik dosyası okunamadı)")
                y -= 1 * cm

    c.save()


# -------------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index(request: Request, db: OrmSession = Depends(get_db)):
    user = current_user(request, db)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "user": user},
    )


@app.get("/register", response_class=HTMLResponse)
def register_get(request: Request):
    return templates.TemplateResponse("register.html", {"request": request, "error": None})


@app.post("/register", response_class=HTMLResponse)
def register_post(
    request: Request,
    full_name: str = Form(...),
    phone: str = Form(...),
    company_name: str = Form(""),
    sector: str = Form(""),
    email: str = Form(...),
    password: str = Form(...),
    db: OrmSession = Depends(get_db),
):
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "error": "Bu e-posta ile zaten bir kullanıcı var."},
        )

    user = User(
        email=email,
        password=password,
        is_admin=False,
        full_name=full_name,
        phone=phone,
        company=company_name,
        sector=sector,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    request.session["user_id"] = user.id
    return RedirectResponse(url="/dashboard", status_code=302)


@app.get("/login", response_class=HTMLResponse)
def login_get(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@app.post("/login", response_class=HTMLResponse)
def login_post(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: OrmSession = Depends(get_db),
):
    user = (
        db.query(User)
        .filter(User.email == email, User.password == password)
        .first()
    )
    if not user:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Geçersiz e-posta veya şifre."},
        )

    request.session["user_id"] = user.id
    return RedirectResponse(url="/dashboard", status_code=302)


@app.get("/admin/login", response_class=HTMLResponse)
def admin_login_get(request: Request):
    return templates.TemplateResponse("admin_login.html", {"request": request, "error": None})


@app.post("/admin/login", response_class=HTMLResponse)
def admin_login_post(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: OrmSession = Depends(get_db),
):
    user = (
        db.query(User)
        .filter(User.email == email, User.password == password, User.is_admin == True)
        .first()
    )
    if not user:
        return templates.TemplateResponse(
            "admin_login.html",
            {"request": request, "error": "Geçersiz yönetici bilgileri."},
        )

    request.session["user_id"] = user.id
    return RedirectResponse(url="/admin/global", status_code=302)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=302)


@app.get("/upload", response_class=HTMLResponse)
def upload_get(request: Request):
    return templates.TemplateResponse(
        "upload.html",
        {
            "request": request,
            "user": None,
        },
    )


@app.post("/upload", response_class=HTMLResponse)
async def upload_post(
    request: Request,
    full_name: str = Form(...),
    company_name: str = Form(...),
    phone: str = Form(...),
    email: str = Form(...),
    sector: str = Form(""),
    file: UploadFile = File(...),
    db: OrmSession = Depends(get_db),
):
    content = await file.read()

    # CSV > Excel fallback
    try:
        df = pd.read_csv(io.BytesIO(content))
        file_type = "csv"
    except Exception:
        try:
            df = pd.read_excel(io.BytesIO(content))
            file_type = "excel"
        except Exception:
            return templates.TemplateResponse(
                "upload.html",
                {
                    "request": request,
                    "user": None,
                    "error": "Dosya okunamadı. Lütfen geçerli bir CSV/Excel dosyası yükleyin.",
                },
            )

    rows, cols = df.shape
    total_cells = int(rows * cols)
    total_missing = int(df.isna().sum().sum())
    quality_score = 100.0
    if total_cells > 0:
        quality_score = max(0.0, 100.0 - (total_missing / total_cells) * 100.0)

    top_missing_col = df.isna().sum().idxmax() if cols > 0 else None
    var_series = df.var(numeric_only=True)
    top_var_col = var_series.idxmax() if not var_series.empty else None

    domain_insights = ["Veri alani belirtilmemis, ozel KPI calismasi onerilir."]
    company_label = company_name or "Firma Belirtilmedi"

    upload = Upload(
        user_id=None,
        file_name=file.filename,
        file_type=file_type,
        company=company_label,
        row_count=rows,
        col_count=cols,
        total_cells=total_cells,
        total_missing=total_missing,
        quality_score=quality_score,
        top_missing_col=top_missing_col,
        top_var_col=top_var_col,
        domain_insights="\n".join(domain_insights),
        contact_name=full_name,
        contact_phone=phone,
        contact_email=email,
        contact_sector=sector,
    )
    db.add(upload)
    db.commit()
    db.refresh(upload)

    # AI analiz
    ai_insights = ai_analyze_dataframe(df, sector=sector)
    ai_summary = ai_insights.get("summary", "")
    ai_risks = ai_insights.get("risks", "")
    ai_features = ai_insights.get("features", "")
    ai_models = ai_insights.get("ml_models", "")
    ai_recommendations = ai_insights.get("recommendations", "")

    upload.ai_summary = ai_summary
    upload.ai_risks = ai_risks
    upload.ai_features = ai_features
    upload.ai_models = ai_models
    upload.ai_recommendations = ai_recommendations
    db.commit()
    db.refresh(upload)

    # Grafikler
    charts_raw = generate_charts(df, upload_id=upload.id)
    chart_cards = build_chart_cards(charts_raw)

    ANALYSIS_CACHE[upload.id] = {
        "file_name": file.filename,
        "file_type": file_type,
        "row_count": rows,
        "col_count": cols,
        "total_cells": total_cells,
        "total_missing": total_missing,
        "quality_score": quality_score,
        "top_missing_col": top_missing_col,
        "top_var_col": top_var_col,
        "domain_insights": domain_insights,
        "charts": chart_cards,
        "company": company_label,
        "ai_summary": ai_summary,
        "ai_risks": ai_risks,
        "ai_features": ai_features,
        "ai_models": ai_models,
        "ai_recommendations": ai_recommendations,
        "contact_name": full_name,
        "contact_phone": phone,
        "contact_email": email,
        "contact_sector": sector,
    }

    request.session["last_upload_id"] = upload.id

    analysis_ctx = {
        "rows": rows,
        "cols": cols,
        "total_cells": total_cells,
        "total_missing": total_missing,
        "quality_score": quality_score,
        "top_missing_col": top_missing_col,
        "top_var_col": top_var_col,
        "domain_insights": domain_insights,
        "charts": chart_cards,
        "ai_summary": ai_summary,
        "ai_risks": ai_risks,
        "ai_features": ai_features,
        "ai_models": ai_models,
        "ai_recommendations": ai_recommendations,
    }

    return templates.TemplateResponse(
        "report.html",
        {
            "request": request,
            "user": None,
            "analysis": analysis_ctx,
            "charts": chart_cards,
            "ai_comment": ai_summary,
            "ai_report": ai_recommendations,
            "company": company_label,
            "file_name": file.filename,
            "file_type": file_type,
            "contact_name": full_name,
            "contact_phone": phone,
            "contact_email": email,
            "contact_sector": sector,
            "upload_id": upload.id,
        },
    )


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, db: OrmSession = Depends(get_db)):
    user = current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    uploads = (
        db.query(Upload)
        .filter(Upload.user_id == user.id)
        .order_by(Upload.created_at.desc())
        .all()
    )

    data = None
    charts: List[Dict[str, Any]] = []

    last_upload_id = request.session.get("last_upload_id")
    if last_upload_id and last_upload_id in ANALYSIS_CACHE:
        cached = ANALYSIS_CACHE[last_upload_id]
        data = {
            "quality_score": cached["quality_score"],
            "total_rows": cached["row_count"],
            "total_columns": cached["col_count"],
            "total_missing": cached["total_missing"],
            "ai_summary": cached.get("ai_summary", ""),
            "ai_details": cached.get("ai_recommendations", ""),
        }
        charts = cached.get("charts", [])

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "user": user,
            "uploads": uploads,
            "data": data,
            "charts": charts,
        },
    )


@app.get("/reports", response_class=HTMLResponse)
def reports(request: Request, db: OrmSession = Depends(get_db)):
    user = current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    uploads = (
        db.query(Upload)
        .filter(Upload.user_id == user.id)
        .order_by(Upload.created_at.desc())
        .all()
    )

    return templates.TemplateResponse(
        "reports.html",
        {
            "request": request,
            "user": user,
            "uploads": uploads,
        },
    )


@app.get("/admin", response_class=HTMLResponse)
def admin_redirect():
    return RedirectResponse(url="/admin/global", status_code=302)


@app.get("/admin/global", response_class=HTMLResponse)
def admin_global(request: Request, db: OrmSession = Depends(get_db)):
    user = current_user(request, db)
    if not user or not user.is_admin:
        return RedirectResponse(url="/admin/login", status_code=302)

    total_users = db.query(User).count()
    total_uploads = db.query(Upload).count()
    last_uploads = (
        db.query(Upload)
        .order_by(Upload.created_at.desc())
        .limit(10)
        .all()
    )

    return templates.TemplateResponse(
        "admin_global.html",
        {
            "request": request,
            "user": user,
            "total_users": total_users,
            "total_uploads": total_uploads,
            "last_uploads": last_uploads,
        },
    )


@app.get("/download_pdf/{upload_id}")
def download_pdf(upload_id: int, db: OrmSession = Depends(get_db)):
    """
    ANALYSIS_CACHE veya DB'deki AI sonuçlarını ve grafik dosyalarını kullanarak PDF raporu indir.
    """
    upload = db.query(Upload).filter(Upload.id == upload_id).first()
    if not upload:
        raise HTTPException(status_code=404, detail="Rapor bulunamadı.")

    cached: Dict[str, Any] = ANALYSIS_CACHE.get(upload_id) or {}

    summary = cached.get("ai_summary", upload.ai_summary or "")
    risks = cached.get("ai_risks", upload.ai_risks or "")
    features = cached.get("ai_features", upload.ai_features or "")
    models = cached.get("ai_models", upload.ai_models or "")
    recs = cached.get("ai_recommendations", upload.ai_recommendations or "")

    chart_files: List[str] = []

    charts_data = cached.get("charts")
    chart_cards: List[Dict[str, Any]] = []

    if isinstance(charts_data, list):
        chart_cards = charts_data
    elif isinstance(charts_data, dict):
        chart_cards = build_chart_cards(charts_data)
    else:
        chart_cards = []

    for ch in chart_cards:
        if not isinstance(ch, dict):
            continue
        url = ch.get("url")
        if not url:
            continue
        fname = os.path.basename(url)
        fpath = os.path.join(CHART_DIR, fname)
        if os.path.exists(fpath):
            chart_files.append(fpath)

    pdf_path = os.path.join(REPORT_DIR, f"rapor_{upload_id}.pdf")

    meta: Optional[Dict[str, Any]] = {
        "company": upload.company or "",
        "contact_name": upload.contact_name or "",
        "contact_email": upload.contact_email or "",
        "contact_phone": upload.contact_phone or "",
        "contact_sector": upload.contact_sector or "",
    }

    generate_pdf_report(
        output_path=pdf_path,
        summary=summary,
        risks=risks,
        features=features,
        models=models,
        recommendations=recs,
        chart_files=chart_files,
        meta=meta,
    )

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"veri_raporu_{upload_id}.pdf",
    )
