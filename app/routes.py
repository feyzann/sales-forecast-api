from __future__ import annotations

from typing import Any, Dict, List, Optional
import requests
import threading
import time

from flask import Blueprint, current_app, jsonify, request

from .auth import extract_token_from_headers, is_authorized
from .services.pipeline import build_pipeline


api_bp = Blueprint("api", __name__)


def _bad_request(message: str, error_code: str = "bad_request"):
    return jsonify({"error": error_code, "message": message}), 400


def _unauthorized(message: str = "Unauthorized"):
    return jsonify({"error": "unauthorized", "message": message}), 401


def _unprocessable(message: str, error_code: str = "unprocessable_entity"):
    return jsonify({"error": error_code, "message": message}), 422


@api_bp.route("/predict", methods=["POST"])
def predict():
    # token = extract_token_from_headers(request)
    # if not is_authorized(token, current_app.config["API_SECRET_TOKENS"], current_app.config["API_KEYS"]):
    #     return _unauthorized("Geçersiz veya eksik API anahtarı.")

    try:
        payload: Dict[str, Any] = request.get_json(force=True, silent=False)  # raise if not JSON
    except Exception:
        return _bad_request("Geçersiz JSON gövdesi.")

    data: Optional[List[Dict[str, Any]]] = payload.get("data")
    prediction_period: Optional[int] = payload.get("prediction_period")
    prediction_frequency: Optional[str] = payload.get("prediction_frequency")
    feature_columns: Optional[List[str]] = payload.get("feature_columns")
    confidence_interval: bool = bool(payload.get("confidence_interval", False))

    if data is None or not isinstance(data, list) or len(data) == 0:
        return _bad_request("'data' zorunludur ve boş olmamalıdır.", "missing_parameter")
    if prediction_period is None or not isinstance(prediction_period, int) or prediction_period <= 0:
        return _bad_request("'prediction_period' zorunludur ve pozitif bir sayı olmalıdır.", "missing_parameter")
    if not prediction_frequency or prediction_frequency not in {"weekly", "monthly"}:
        return _bad_request("'prediction_frequency' zorunludur ve 'weekly'/'monthly' olmalıdır.", "missing_parameter")

    # Callback benzeri parametre kontrolü 
    # burayı sor?
    callback_url = None
    for key, value in payload.items():
        if key.lower() in ['callback', 'webhook', 'notify', 'async', 'background']:
            callback_url = value
            break

    if callback_url:
        # Callback varsa hemen 200 dön
        request_id = "req_" + str(int(time.time()))
        response_data = {
            "success": True,
            "message": "İstek alındı, işlem arka planda devam ediyor",
            "request_id": request_id,
            "status": "processing",
            "callback_url": callback_url
        }
        
        # Arka planda callback'e sonucu gönder
        def send_callback():
            try:
                # API key header'ı ile callback'e POST
                headers = {
                    "X-API-Key": current_app.config.get("CALLBACK_API_KEY", ""),
                    "Content-Type": "application/json",
                    "X-Request-ID": request_id
                }
                
                # Tahmin işlemini çalıştır
                pipeline = build_pipeline(
                    prediction_frequency=prediction_frequency,
                    aggregation_level=prediction_frequency,  # Otomatik olarak prediction_frequency ile aynı
                    prediction_period=prediction_period,
                    feature_columns=feature_columns or [],
                    return_confidence=confidence_interval,
                    min_data_points=current_app.config["MIN_DATA_POINTS"],
                )
                result = pipeline.run(data)
                
                # Sonucu callback URL'e gönder
                requests.post(callback_url, json=result, headers=headers, timeout=30)
                
            except Exception as e:
                # Hata durumunda da callback'e bilgi gönder
                error_data = {
                    "error": "callback_failed", 
                    "message": str(e),
                    "request_id": request_id,
                    "status": "failed"
                }
                try:
                    requests.post(callback_url, json=error_data, headers=headers, timeout=30)
                except:
                    pass  # Callback'e hata gönderilemezse sessizce devam et
        
        # Arka planda çalıştır
        threading.Thread(target=send_callback, daemon=True).start()
        
        return jsonify(response_data), 200

    # Callback yoksa normal senkron işlem
    # Build and run pipeline
    pipeline = build_pipeline(
        prediction_frequency=prediction_frequency,
        aggregation_level=prediction_frequency,  # Otomatik olarak prediction_frequency ile aynı
        prediction_period=prediction_period,
        feature_columns=feature_columns or [],
        return_confidence=confidence_interval,
        min_data_points=current_app.config["MIN_DATA_POINTS"],
    )

    try:
        result = pipeline.run(data)
    except ValueError as ve:
        return _unprocessable(str(ve), "insufficient_data")
    except Exception as ex:
        return jsonify({"error": "internal_error", "message": str(ex)}), 500

    return jsonify(result), 200
