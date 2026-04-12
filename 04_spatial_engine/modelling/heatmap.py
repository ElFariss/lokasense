#!/usr/bin/env python3
"""
04_spatial_engine/modelling/heatmap.py
Generate Folium choropleth heatmap from opportunity scores.
"""
import json
import pandas as pd
import folium
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
LOG_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_marker_map(scores_df):
    """Create a simple marker-based map (no GeoJSON required)."""
    # Load gazetteer for coordinates
    gaz_dir = DATA_DIR / "geospatial" / "Wilayah-Administratif-Indonesia" / "csv"
    gaz_files = list(gaz_dir.glob("*.csv"))
    
    # Approximate city centers
    city_coords = {
        "Malang": (-7.977, 112.634),
        "Surabaya": (-7.250, 112.751),
        "Yogyakarta": (-7.797, 110.361),
        "Bandung": (-6.914, 107.609),
        "Semarang": (-6.966, 110.419),
    }

    # Map center
    m = folium.Map(location=[-7.5, 110.4], zoom_start=7, tiles="cartodbpositron")

    # Color mapping
    color_map = {"green": "#2ecc71", "yellow": "#f39c12", "red": "#e74c3c"}

    for _, row in scores_df.iterrows():
        city = row.get("city", "")
        lat = row.get("resolved_lat")
        lon = row.get("resolved_lng")
        if pd.isna(lat) or pd.isna(lon):
            if city not in city_coords:
                continue
            lat, lon = city_coords[city]
            # Offset markers slightly per kecamatan to avoid overlap
            import hashlib

            offset = int(hashlib.md5(str(row.get("kecamatan", "")).encode()).hexdigest()[:4], 16) / 65535
            lat += (offset - 0.5) * 0.05
            lon += (offset - 0.5) * 0.05

        score = row.get("opportunity_score", 0.5)
        color = row.get("color", "yellow")
        kec = row.get("kecamatan", "N/A")
        biz = row.get("business_type", "N/A")
        label = row.get("label", "")

        popup_html = f"""
        <b>{kec}, {city}</b><br>
        Business: {biz}<br>
        Score: <b>{score:.2f}</b><br>
        Status: <span style='color:{color_map.get(color, "#999")}'><b>{label}</b></span><br>
        Signals: {row.get('total_signals', 0)}<br>
        Avg age: {row.get('avg_age_days', 0)} days<br>
        Location source: {row.get('resolution_source', 'fallback')}
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=max(5, score * 20),
            color=color_map.get(color, "#999"),
            fill=True,
            fill_color=color_map.get(color, "#999"),
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{kec}: {score:.2f}",
        ).add_to(m)

    # Legend
    legend_html = '''
    <div style="position:fixed; bottom:50px; left:50px; z-index:1000; 
         background:white; padding:10px; border-radius:5px; box-shadow: 0 0 5px rgba(0,0,0,0.3);">
    <b>LokaSense — Opportunity Map</b><br>
    <span style="color:#2ecc71">●</span> Strong Opportunity (≥0.65)<br>
    <span style="color:#f39c12">●</span> Moderate (0.40-0.65)<br>
    <span style="color:#e74c3c">●</span> Saturated/Risky (<0.40)<br>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


def main():
    print("=" * 60)
    print(" Heatmap Generation — LokaSense")
    print("=" * 60)

    scores_file = LOG_DIR / "opportunity_scores.csv"
    if not scores_file.exists():
        print("No scores found. Run scoring.py first.")
        return

    scores_df = pd.read_csv(scores_file)
    print(f"  Loaded {len(scores_df)} scored areas")

    m = create_marker_map(scores_df)
    output_file = OUTPUT_DIR / "lokasense_heatmap.html"
    m.save(str(output_file))

    print(f"\nHeatmap saved to: {output_file}")
    print(f"  Open in browser to see the interactive map!")


if __name__ == "__main__":
    main()
