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

# -------------------------------------------------------------------
# Ortam değişkenleri / yollar
# -------------------------------------------------------------------

load_dotenv()

OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()

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
