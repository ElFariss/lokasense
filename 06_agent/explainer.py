from __future__ import annotations

import ast

import pandas as pd

SIGNAL_LABELS_ID = {
    "DEMAND_UNMET": "permintaan belum terpenuhi",
    "DEMAND_PRESENT": "permintaan ada",
    "TREND": "tren",
    "COMPETITION_HIGH": "persaingan tinggi",
    "COMPLAINT": "keluhan",
    "SUPPLY_SIGNAL": "pasokan",
    "NEUTRAL": "netral",
}

STATUS_LABELS_ID = {
    "Strong Opportunity": "Peluang Tinggi",
    "Moderate Opportunity": "Peluang Sedang",
    "Saturated / Risky": "Jenuh / Berisiko",
}


def _coerce_breakdown(value) -> dict[str, float]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def _recommendation(dominant_signal: str, business_type: str) -> str:
    mapping = {
        "DEMAND_UNMET": f"Area ini menunjukkan gap kebutuhan untuk {business_type} yang belum terisi.",
        "DEMAND_PRESENT": f"Permintaan untuk {business_type} sudah ada, tetapi diferensiasi tetap penting.",
        "TREND": f"Percakapan tentang {business_type} sedang bergerak cepat dan cocok untuk uji pasar.",
        "COMPETITION_HIGH": f"Area ini padat pesaing untuk {business_type}; masuk hanya jika proposisi nilainya kuat.",
        "COMPLAINT": f"Keluhan konsumen membuka peluang perbaikan kualitas pada bisnis {business_type}.",
        "SUPPLY_SIGNAL": f"Pasokan {business_type} sudah terlihat aktif sehingga margin peluang lebih tipis.",
        "NEUTRAL": f"Sinyal untuk {business_type} masih tipis sehingga area ini perlu divalidasi ulang.",
    }
    return mapping.get(dominant_signal, f"Gunakan area ini sebagai pembanding untuk strategi {business_type}.")


def generate_explanation(
    intent,
    scores_df: pd.DataFrame,
    lime_data: dict[str, list[dict[str, object]]] | None = None,
    top_n: int = 5,
    source_mode: str = "airgap",
) -> str:
    if scores_df.empty:
        return f"Belum ada skor peluang yang cukup untuk {intent.business_type} di {intent.city}."

    lime_data = lime_data or {}
    ranked = scores_df.sort_values("opportunity_score", ascending=False).head(top_n)
    source_label = "offline (airgap)" if source_mode == "airgap" else "live publik"

    lines = [
        f"=== Analisis Peluang Bisnis: {intent.business_type} di {intent.city} ===",
        f"Data dianalisis secara {source_label} dari {int(ranked['total_signals'].sum())} sinyal pada {ranked['kecamatan'].nunique()} kecamatan.",
        "",
    ]

    for _, row in ranked.iterrows():
        breakdown = _coerce_breakdown(row.get("signal_breakdown", {}))
        total_signals = max(int(row.get("total_signals", 0)), 1)
        dominant_signal = "NEUTRAL"
        dominant_count = 0.0
        if breakdown:
            dominant_signal, dominant_count = max(breakdown.items(), key=lambda item: float(item[1]))

        status = STATUS_LABELS_ID.get(str(row.get("label", "")), str(row.get("label", "")))
        lines.append(
            f"{row.get('kecamatan', 'Unknown')} - skor {float(row.get('opportunity_score', 0.0)):.2f} - {status}"
        )
        lines.append(
            f"  Total sinyal: {total_signals}. Dominan: {SIGNAL_LABELS_ID.get(dominant_signal, dominant_signal)} "
            f"({float(dominant_count):.1f})."
        )
        lines.append(
            f"  Rekomendasi: {_recommendation(dominant_signal, intent.business_type)}"
        )

        if row.get("franchise_ratio", 0):
            lines.append(f"  Rasio waralaba terpantau: {float(row.get('franchise_ratio', 0)):.2f}.")

        top_tokens = lime_data.get(str(row.get("kecamatan", "")), [])[:3]
        if top_tokens:
            token_summary = ", ".join(
                f"{token['token']} ({float(token['weight']):+.3f})"
                for token in top_tokens
            )
            lines.append(f"  Sinyal kata utama: {token_summary}.")
        lines.append("")

    return "\n".join(lines).strip()
