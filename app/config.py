import os
from typing import List, Set
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


class Config:
    """Application configuration (prod-safe defaults).

    Notlar
    ------
    - API token'ları virgülle ayrılmış olarak verilebilir.
    - Frekans: sadece 'weekly' ve 'monthly' desteklenir.
    - Haftalık ankraj: Pazartesi (W-MON).
    - MIN_DATA_POINTS, **toplama SONRASI** minimum nokta sayısıdır.
    """

    # Güvenlik: Varsayılanı boş bırak (prod'da yanlışlıkla örnek token'la açılmasın)
    API_SECRET_TOKENS: List[str] = [
        t.strip()
        for t in os.getenv("API_SECRET_TOKENS", "").split(",")
        if t.strip()
    ]
    API_KEYS: List[str] = [
        t.strip()
        for t in os.getenv("API_KEYS", "").split(",")
        if t.strip()
    ]

    # Tahmin frekansları
    ALLOWED_FREQUENCIES: Set[str] = {"weekly", "monthly"}
    DEFAULT_PREDICTION_FREQUENCY: str = os.getenv("DEFAULT_PREDICTION_FREQUENCY", "weekly")

    # Zaman kuralları (resample + forecast için aynı ankraj)
    WEEKLY_RULE: str = os.getenv("WEEKLY_RULE", "W-MON")  # Pazartesi
    MONTHLY_RULE: str = os.getenv("MONTHLY_RULE", "MS")   # Month Start

    # Veri ve performans
    MIN_DATA_POINTS: int = int(os.getenv("MIN_DATA_POINTS", "30"))  # post-aggregation
    REQUEST_TIMEOUT_SECONDS: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "300"))

    # Çıkış davranışları
    RETURN_CONFIDENCE_DEFAULT: bool = _env_bool("RETURN_CONFIDENCE_DEFAULT", False) # burayı sor?
    NON_NEGATIVE_PREDICTIONS: bool = _env_bool("NON_NEGATIVE_PREDICTIONS", True) # burayı sor?
    BACKTEST_POINTS: int = int(os.getenv("BACKTEST_POINTS", "12"))

    # Callback konfigürasyonu
    CALLBACK_API_KEY: str = os.getenv("CALLBACK_API_KEY", "")  # Callback'e gönderilecek API key, burayı sor?
    CALLBACK_TIMEOUT: int = int(os.getenv("CALLBACK_TIMEOUT", "30"))  # Callback timeout saniye, burayı sor?

    @classmethod
    def freq_to_rule(cls, freq: str) -> str:
        if freq == "weekly":
            return cls.WEEKLY_RULE
        if freq == "monthly":
            return cls.MONTHLY_RULE
        raise ValueError(f"Unsupported frequency: {freq}")
