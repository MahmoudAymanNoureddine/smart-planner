"""
Smart Environmental Planner — PhD Platform v5.0 (Final)
========================================================
Phase 1: Prototype | Phase 2: AI & GIS | Phase 3: Smart Platform
+ Chat AI Assistant | + Professional Maps | + All PDF Elements
"""
import streamlit as st
import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MiniMap, MeasureControl, Fullscreen
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation
import plotly.graph_objects as go
import plotly.express as px
import random
from datetime import datetime, timedelta

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

st.set_page_config(page_title="PhD Smart Planner", page_icon="🎓", layout="wide")

# ============================================================
# 1. قواعد البيانات
# ============================================================
NATIVE_PLANTS_DB = {
    "Acacia (السنط)": {"soil": ["Sandy", "Loamy"], "water": "Low"},
    "Ziziphus (السدر)": {"soil": ["Sandy", "Loamy", "Clay"], "water": "Low"},
    "Olea (الزيتون)": {"soil": ["Loamy", "Clay"], "water": "Low"},
    "Ficus (اللبخ)": {"soil": ["Loamy", "Clay"], "water": "Medium"},
    "Bougainvillea (الجهنمية)": {"soil": ["Sandy", "Loamy"], "water": "Low"},
    "Lantana (اللانتانا)": {"soil": ["Sandy", "Loamy"], "water": "Low"},
    "Dodonaea (الدودونيا)": {"soil": ["Sandy"], "water": "Low"},
}

PALETTE = {
    'building': '#64748b', 'building_border': '#1e293b',
    'green': '#10b981', 'green_border': '#047857',
    'water': '#0ea5e9', 'water_border': '#0c4a6e',
    'solar_high': '#fbbf24', 'solar_mid': '#f59e0b',
    'tree': '#059669',
    'bin_organic': '#92400e', 'bin_plastic': '#1d4ed8',
    'bin_paper': '#ca8a04', 'bin_glass': '#16a34a', 'bin_general': '#4b5563',
    'light': '#facc15', 'irrigation': '#0891b2',
    'corridor': '#a855f7', 'path': '#f97316',
}

KB = {
    'water': {'irrigation_benchmark': 12, 'peak': 15, 'unit': 'L/m²/day',
              'source': 'Shammas 1991 — Riyadh Parks',
              'notes': 'معدل استهلاك الري الموصى به للمساحات الخضراء العامة. في موسم الذروة يصل إلى 15 L/m²/day.'},
    'energy': {'good_benchmark': 3, 'typical_benchmark': 7, 'unit': 'kWh/m²/yr',
               'source': 'BBP 2025 — REEB Benchmark',
               'notes': '3 = ممارسة جيدة، 7 = ممارسة نموذجية.'},
    'waste': {'visitor_waste': 0.05, 'green_waste_min': 1.5, 'green_waste_max': 2.0,
              'recycling_target': 70, 'contamination_max': 30,
              'source': 'Mukhtar 2016 + Song 2025 + Cooley 2018',
              'notes': '0.05 kg/زائر/يوم، 1.5-2.0 kg/m²/yr للنفايات الخضراء. التدوير المستهدف 70%.'},
    'carbon': {'seq_min': 5.61, 'seq_max': 7.24, 'grid_factor': 0.45,
               'source': 'Zhou 2025 + Basaly 2019',
               'notes': 'الكربون المتسلسل: 5.61-7.24 kg CO2e/m²/yr.'},
    'biodiversity': {'max_species': 10, 'max_genus': 20, 'max_family': 30,
                     'tree_density': 150, 'soil_small': 17, 'soil_medium': 28.3, 'soil_large': 42.5,
                     'source': 'Santamour 1990 + Lindén 2020 + DeepRoot',
                     'notes': 'قاعدة 10/20/30. كثافة 150 شجرة/هكتار.'},
    'soil': {'loamy_density_max': 1.4, 'sandy_density_max': 1.6,
             'source': 'USDA NRCS 2023 + FHWA 2015',
             'notes': 'التربة الطميية: كثافة <1.4 g/cm³. الرملية: <1.6.'},
    'human': {'source': 'دراسة السلوك البشري - PDF',
              'notes': 'السلوك البشري يؤثر على كفاءة الإدارة البيئية.'},
    'climate': {'source': 'PDF - North Orientation',
                'notes': 'اتجاه الشمال مهم لتحليل الظل والطاقة الشمسية.'},
}

# ============================================================
# 2. Session State
# ============================================================
defaults = {
    'site_area_m2': 10000.0, 'soil_type': "Loamy", 'visitors_per_day': 500,
    'climate_temp': 30.0, 'rainfall_mm': 50.0, 'irrigated_area_pct': 60,
    'irrigation_system': "Drip", 'water_source': "Treated Wastewater",
    'lighting_hours': 10, 'solar_pct': 30, 'has_pumps': True,
    'bins_capacity': 40, 'tree_species_count': 5, 'trees_count': 100,
    'has_composting': False, 'flow_type': "Linear",
    'data_quality': {"soil": False}, 'iot_running': False,
    'ai_recommendations': "", 'my_location': None, 'site_stats': {},
    'chat_history': None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# 3. الحسابات
# ============================================================
def compute_all_metrics(override=None):
    o = override or {}
    area = o.get('area', st.session_state['site_area_m2'])
    temp = o.get('temp', st.session_state['climate_temp'])
    rain = o.get('rain', st.session_state['rainfall_mm'])
    irr_pct = o.get('irr_pct', st.session_state['irrigated_area_pct'])
    irr_sys = o.get('irr_sys', st.session_state['irrigation_system'])
    water_src = o.get('water_src', st.session_state['water_source'])
    light_hrs = o.get('light', st.session_state['lighting_hours'])
    solar_pct = o.get('solar', st.session_state['solar_pct'])
    pumps = o.get('pumps', st.session_state['has_pumps'])
    visitors = o.get('visitors', st.session_state['visitors_per_day'])
    species = o.get('species', st.session_state['tree_species_count'])
    trees = o.get('trees', st.session_state['trees_count'])
    compost = o.get('compost', st.session_state['has_composting'])

    irr_area = area * (irr_pct / 100)
    adj_demand = 12 * max(0.5, temp / 30) * max(0.7, 1 - (rain / 200))
    eff = {"Drip": 0.9, "Sprinkler": 0.75, "Manual": 0.6}[irr_sys]
    daily_water = (irr_area * adj_demand) / eff
    annual_m3 = daily_water * 365 / 1000
    water_m2 = daily_water / area if area else 0
    light_kwh = (area * 1.5) * light_hrs * 365 / 1000
    pump_kwh = (area * 0.5) * 4 * 365 / 1000 if pumps else 0
    total_kwh = light_kwh + pump_kwh
    energy_m2 = total_kwh / area if area else 0
    carbon_kg = total_kwh * (1 - solar_pct / 100) * 0.45
    carbon_seq = irr_area * 6.4
    net_carbon = carbon_kg - carbon_seq
    waste_kg = visitors * 0.05
    recycling = 0.35 if compost else 0.10
    native_ratio = min(1.0, species / 10)

    return {'area': area, 'water_m2': water_m2, 'annual_m3': annual_m3, 'daily_water': daily_water,
            'water_src': water_src, 'irr_sys': irr_sys, 'irr_area': irr_area,
            'energy_m2': energy_m2, 'total_kwh': total_kwh, 'solar': solar_pct,
            'carbon_kg': carbon_kg, 'carbon_seq': carbon_seq, 'net_carbon': net_carbon,
            'waste_kg': waste_kg, 'recycling': recycling, 'species': species,
            'trees': trees, 'native_ratio': native_ratio, 'compost': compost, 'visitors': visitors}

def compute_scores(m):
    def cl(v): return max(0, min(100, v))
    w = 100 if m['water_m2'] <= 8 else cl(100 - (m['water_m2'] - 8) * 8)
    if m['water_src'] == "Treated Wastewater": w = cl(w + 8)
    e = 100 if m['energy_m2'] <= 3 else cl(100 - (m['energy_m2'] - 3) * 15)
    e = cl(e + m['solar'] * 0.1)
    ws = cl((m['recycling'] / 0.7) * 100)
    c = 100 if m['net_carbon'] <= 0 else cl(100 - (m['net_carbon'] / max(m['carbon_kg'], 1)) * 100)
    b = cl((m['species'] / 10) * 100)
    r = np.mean([w, e, ws, c, b])
    return {'water': round(w, 1), 'energy': round(e, 1), 'waste': round(ws, 1),
            'carbon': round(c, 1), 'biodiversity': round(b, 1), 'resource': round(r, 1)}

def resolve_any_location(q_type, q_value):
    try:
        if q_type == "📍 موقعي الحالي":
            return st.session_state.get('my_location')
        elif q_type == "اسم مكان":
            if not q_value.strip(): return None
            try:
                r = ox.geocode(q_value)
                if r: return r
            except: pass
            try:
                g = ox.geocode_to_gdf(q_value)
                if not g.empty:
                    c = g.geometry.unary_union.centroid
                    return (c.y, c.x)
            except: pass
            return None
        else:
            parts = q_value.split(",")
            if len(parts) == 2:
                lat, lon = float(parts[0].strip()), float(parts[1].strip())
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return (lat, lon)
    except: return None
    return None

@st.cache_data(show_spinner="📥 تحميل بيانات GIS...", ttl=3600)
def fetch_site_data_enhanced(center, radius):
    tags = {"building": True, "natural": ["water", "wood", "scrub", "grassland"],
            "leisure": ["park", "garden", "pitch", "playground"],
            "landuse": ["forest", "grass", "meadow", "recreation_ground"],
            "highway": ["footway", "pedestrian", "path", "cycleway"]}
    try: return ox.features_from_point(center, tags=tags, dist=radius)
    except: return gpd.GeoDataFrame()

def compute_actual_stats(gdf):
    stats = {'buildings_count': 0, 'green_count': 0, 'water_count': 0, 'roads_count': 0,
             'buildings_area': 0, 'green_area': 0, 'water_area': 0}
    if gdf.empty: return stats
    try:
        if 'building' in gdf.columns:
            b = gdf[gdf['building'].notna()]
            stats['buildings_count'] = len(b)
            stats['buildings_area'] = b.to_crs(3857).geometry.area.sum()
        green_mask = pd.Series(False, index=gdf.index)
        if 'leisure' in gdf.columns:
            green_mask |= gdf['leisure'].isin(['park', 'garden', 'pitch']).fillna(False)
        if 'landuse' in gdf.columns:
            green_mask |= gdf['landuse'].isin(['forest', 'grass', 'meadow']).fillna(False)
        if 'natural' in gdf.columns:
            green_mask |= gdf['natural'].isin(['wood', 'scrub', 'grassland']).fillna(False)
        g = gdf[green_mask]
        stats['green_count'] = len(g)
        stats['green_area'] = g.to_crs(3857).geometry.area.sum() if not g.empty else 0
        if 'natural' in gdf.columns:
            w = gdf[gdf['natural'] == 'water']
            stats['water_count'] = len(w)
            stats['water_area'] = w.to_crs(3857).geometry.area.sum() if not w.empty else 0
        if 'highway' in gdf.columns:
            stats['roads_count'] = len(gdf[gdf['highway'].notna()])
    except: pass
    return stats

# ============================================================
# 4. مكونات بصرية احترافية
# ============================================================
def styled_popup(title, color, items, icon=""):
    rows = ''.join([
        f'<tr style="border-bottom:1px solid #f1f5f9;">'
        f'<td style="color:#64748b;padding:8px 12px;font-size:12px;text-align:right;">{k}</td>'
        f'<td style="font-weight:700;padding:8px 12px;font-size:13px;color:#0f172a;text-align:left;">{v}</td></tr>'
        for k, v in items
    ])
    return folium.Popup(f'''
    <div style="font-family:'Segoe UI',Tahoma,sans-serif;min-width:250px;max-width:340px;
        border-radius:14px;overflow:hidden;box-shadow:0 8px 32px rgba(0,0,0,0.25);border:1px solid #e2e8f0;">
        <div style="background:linear-gradient(135deg,{color} 0%,{color}cc 100%);
            color:white;padding:14px 18px;font-weight:700;font-size:14px;
            display:flex;align-items:center;gap:10px;">
            <span style="font-size:20px;">{icon}</span><span>{title}</span>
        </div>
        <table style="width:100%;border-collapse:collapse;background:white;direction:rtl;">{rows}</table>
    </div>''', max_width=350)

def pin_marker(lat, lon, color, emoji, tooltip="", popup=None, size=32):
    html = f'''<div style="position:relative;filter:drop-shadow(0 4px 8px rgba(0,0,0,0.45));
        transition:transform 0.2s;" onmouseover="this.style.transform='scale(1.2)'"
        onmouseout="this.style.transform='scale(1)'">
        <div style="background:linear-gradient(135deg,{color} 0%,{color}dd 100%);
            width:{size}px;height:{size}px;border-radius:50% 50% 50% 0;transform:rotate(-45deg);
            border:3px solid white;box-shadow:0 3px 8px rgba(0,0,0,0.35);
            display:flex;align-items:center;justify-content:center;">
            <div style="transform:rotate(45deg);font-size:{int(size*0.5)}px;line-height:1;">
                {emoji}</div></div></div>'''
    return folium.Marker(
        location=[lat, lon],
        icon=folium.DivIcon(html=html, icon_size=(size, size), icon_anchor=(size//2, size)),
        tooltip=folium.Tooltip(tooltip, sticky=True) if tooltip else None,
        popup=popup)

def dot_marker(lat, lon, color, emoji="", tooltip="", popup=None, size=24):
    html = f'''<div style="background:linear-gradient(135deg,{color} 0%,{color}dd 100%);
        width:{size}px;height:{size}px;border-radius:50%;border:2.5px solid white;
        box-shadow:0 2px 10px rgba(0,0,0,0.35);display:flex;align-items:center;
        justify-content:center;font-size:{int(size*0.55)}px;line-height:1;
        transition:transform 0.2s;" onmouseover="this.style.transform='scale(1.3)'"
        onmouseout="this.style.transform='scale(1)'">{emoji}</div>'''
    return folium.Marker(
        location=[lat, lon],
        icon=folium.DivIcon(html=html, icon_size=(size, size), icon_anchor=(size//2, size//2)),
        tooltip=folium.Tooltip(tooltip, sticky=True) if tooltip else None,
        popup=popup)

def build_legend(overall_score, status, color):
    return f'''
    <div style="position:fixed;bottom:25px;left:25px;width:320px;
        background:rgba(255,255,255,0.92);backdrop-filter:blur(20px);
        border-radius:20px;z-index:9999;font-family:'Segoe UI',Tahoma,sans-serif;
        box-shadow:0 20px 60px rgba(0,0,0,0.3),0 0 0 1px rgba(255,255,255,0.7);
        overflow:hidden;border:1px solid rgba(255,255,255,0.6);direction:rtl;">
        <div style="background:linear-gradient(135deg,#0f172a 0%,#334155 60%,#475569 100%);
            color:white;padding:16px 20px;display:flex;align-items:center;gap:12px;">
            <div style="font-size:22px;">🗺️</div>
            <div><div style="font-weight:700;font-size:15px;">دليل الخريطة</div>
            <div style="font-size:10px;opacity:0.7;">Interactive Map Legend</div></div>
        </div>
        <div style="padding:16px 20px;">
            <div style="font-size:10px;font-weight:700;color:#94a3b8;letter-spacing:1.2px;
                margin-bottom:10px;">بيانات موجودة</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:16px;">
                <div style="display:flex;align-items:center;gap:8px;font-size:12px;color:#334155;">
                    <div style="width:16px;height:16px;background:{PALETTE['building']};border-radius:4px;"></div>مبانٍ</div>
                <div style="display:flex;align-items:center;gap:8px;font-size:12px;color:#334155;">
                    <div style="width:16px;height:16px;background:{PALETTE['green']};border-radius:4px;"></div>خضرة</div>
                <div style="display:flex;align-items:center;gap:8px;font-size:12px;color:#334155;">
                    <div style="width:16px;height:16px;background:{PALETTE['water']};border-radius:4px;"></div>مياه</div>
                <div style="display:flex;align-items:center;gap:8px;font-size:12px;color:#334155;">
                    <div style="width:16px;height:3px;background:{PALETTE['path']};"></div>مسارات</div>
            </div>
            <div style="font-size:10px;font-weight:700;color:#94a3b8;letter-spacing:1.2px;
                margin-bottom:10px;">عناصر مقترحة</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:16px;">
                <div style="display:flex;align-items:center;gap:8px;font-size:12px;color:#334155;">
                    <div style="width:16px;height:16px;background:{PALETTE['solar_high']};border-radius:50%;"></div>سطح شمسي</div>
                <div style="display:flex;align-items:center;gap:8px;font-size:12px;color:#334155;">
                    <div style="width:16px;height:16px;background:{PALETTE['tree']};border-radius:50%;"></div>شجرة</div>
                <div style="display:flex;align-items:center;gap:8px;font-size:12px;color:#334155;">
                    <div style="width:16px;height:16px;background:{PALETTE['light']};border-radius:50%;"></div>إنارة LED</div>
                <div style="display:flex;align-items:center;gap:8px;font-size:12px;color:#334155;">
                    <div style="width:16px;height:16px;background:{PALETTE['irrigation']};border-radius:50%;"></div>نقطة ري</div>
            </div>
            <div style="font-size:10px;font-weight:700;color:#94a3b8;letter-spacing:1.2px;
                margin-bottom:10px;">أنواع السلال (5)</div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-bottom:18px;">
                <div style="display:flex;align-items:center;gap:4px;font-size:11px;color:#334155;">
                    <div style="width:12px;height:12px;background:{PALETTE['bin_organic']};border-radius:50%;"></div>عضوي</div>
                <div style="display:flex;align-items:center;gap:4px;font-size:11px;color:#334155;">
                    <div style="width:12px;height:12px;background:{PALETTE['bin_plastic']};border-radius:50%;"></div>بلاستيك</div>
                <div style="display:flex;align-items:center;gap:4px;font-size:11px;color:#334155;">
                    <div style="width:12px;height:12px;background:{PALETTE['bin_paper']};border-radius:50%;"></div>ورق</div>
                <div style="display:flex;align-items:center;gap:4px;font-size:11px;color:#334155;">
                    <div style="width:12px;height:12px;background:{PALETTE['bin_glass']};border-radius:50%;"></div>زجاج</div>
                <div style="display:flex;align-items:center;gap:4px;font-size:11px;color:#334155;">
                    <div style="width:12px;height:12px;background:{PALETTE['bin_general']};border-radius:50%;"></div>عام</div>
            </div>
            <div style="background:linear-gradient(135deg,{color} 0%,{color}dd 100%);
                padding:12px 16px;border-radius:12px;color:white;box-shadow:0 4px 16px {color}55;">
                <div style="font-size:10px;opacity:0.9;margin-bottom:6px;">أداء الموقع العام</div>
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-size:24px;font-weight:800;">{overall_score}%</span>
                    <span style="font-size:11px;background:rgba(255,255,255,0.3);
                        padding:4px 12px;border-radius:20px;font-weight:600;">{status}</span>
                </div>
            </div>
        </div>
    </div>'''

def build_stats_panel(stats):
    return f'''
    <div style="position:fixed;top:80px;left:25px;background:rgba(255,255,255,0.92);
        backdrop-filter:blur(20px);border-radius:18px;z-index:9999;
        font-family:'Segoe UI',Tahoma,sans-serif;
        box-shadow:0 12px 40px rgba(0,0,0,0.25);padding:16px 18px;
        border:1px solid rgba(255,255,255,0.6);min-width:260px;direction:rtl;">
        <div style="font-size:11px;font-weight:700;color:#64748b;letter-spacing:1.2px;
            margin-bottom:12px;">📊 إحصائيات المنطقة</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
            <div style="background:linear-gradient(135deg,#f1f5f9 0%,#e2e8f0 100%);
                padding:12px;border-radius:12px;text-align:center;">
                <div style="font-size:24px;font-weight:800;color:#334155;">{stats.get('buildings_count', 0)}</div>
                <div style="font-size:10px;color:#64748b;">🏢 مبانٍ</div></div>
            <div style="background:linear-gradient(135deg,#d1fae5 0%,#a7f3d0 100%);
                padding:12px;border-radius:12px;text-align:center;">
                <div style="font-size:24px;font-weight:800;color:#047857;">{stats.get('green_count', 0)}</div>
                <div style="font-size:10px;color:#065f46;">🌳 خضرة</div></div>
            <div style="background:linear-gradient(135deg,#dbeafe 0%,#bfdbfe 100%);
                padding:12px;border-radius:12px;text-align:center;">
                <div style="font-size:24px;font-weight:800;color:#1e40af;">{stats.get('water_count', 0)}</div>
                <div style="font-size:10px;color:#1e3a8a;">💧 مياه</div></div>
            <div style="background:linear-gradient(135deg,#fed7aa 0%,#fdba74 100%);
                padding:12px;border-radius:12px;text-align:center;">
                <div style="font-size:24px;font-weight:800;color:#9a3412;">{stats.get('roads_count', 0)}</div>
                <div style="font-size:10px;color:#7c2d12;">🛣️ مسارات</div></div>
        </div>
    </div>'''

# ============================================================
# 5. الخريطة الاحترافية
# ============================================================
def build_phd_map(center, radius, gdf, metrics, scores):
    m = folium.Map(location=list(center), zoom_start=16, tiles=None,
        control_scale=True, prefer_canvas=True, zoom_control='topright')

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="© Esri", name="🛰️ قمر صناعي", show=True).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        attr="© Esri Labels", name="📝 أسماء الشوارع", overlay=True, show=True).add_to(m)
    folium.TileLayer(tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="© OSM © CartoDB", name="🎨 خريطة نظيفة", show=False).add_to(m)
    folium.TileLayer(tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="© OSM © CartoDB", name="🌙 خريطة داكنة", show=False).add_to(m)

    fg_existing = folium.FeatureGroup(name="🏙️ البيانات الحالية", show=True)
    fg_heat = folium.FeatureGroup(name="🔥 كثافة المباني", show=False)
    fg_solar = folium.FeatureGroup(name="☀️ أسطح شمسية", show=True)
    fg_roads = folium.FeatureGroup(name="🛣️ مسارات", show=True)
    fg_trees = folium.FeatureGroup(name="🌳 أشجار مقترحة", show=True)
    fg_bins = folium.FeatureGroup(name="🗑️ سلال نفايات", show=True)
    fg_lights = folium.FeatureGroup(name="💡 إنارة LED", show=True)
    fg_water = folium.FeatureGroup(name="💧 نقاط ري", show=True)
    fg_corridors = folium.FeatureGroup(name="🦋 ممرات حيوية", show=False)

    delta_lat = radius / 111320
    delta_lon = radius / (111320 * np.cos(np.radians(center[0])))
    bounds = [[center[0]-delta_lat, center[1]-delta_lon],
              [center[0]+delta_lat, center[1]+delta_lon]]

    overall = scores.get('resource', 50)
    perf_color = '#10b981' if overall >= 80 else '#fbbf24' if overall >= 60 else '#f97316' if overall >= 40 else '#ef4444'
    status_txt = "ممتاز" if overall >= 80 else "جيد" if overall >= 60 else "متوسط" if overall >= 40 else "يحتاج تحسين"

    folium.Rectangle(bounds=bounds, color=perf_color, weight=2, fill=True,
        fillOpacity=0.03, dash_array='8,8').add_to(m)

    road_points = []
    if not gdf.empty and 'highway' in gdf.columns:
        roads = gdf[gdf['highway'].notna()]
        for geom in roads.geometry:
            if geom.geom_type == 'LineString':
                try:
                    for d in np.linspace(0, geom.length, 6):
                        p = geom.interpolate(d)
                        road_points.append((p.y, p.x))
                except: pass

    green_centroids = []
    # المباني
    if not gdf.empty and 'building' in gdf.columns:
        b = gdf[gdf['building'].notna()]
        try: b_4326 = b.to_crs(4326)
        except: b_4326 = b
        for _, row in b_4326.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty: continue
            try: area_m2 = gpd.GeoSeries([geom], crs=4326).to_crs(3857).area.iloc[0]
            except: area_m2 = 100
            try: c = geom.representative_point(); cx, cy = c.y, c.x
            except: continue
            if geom.geom_type == 'Polygon':
                coords = [(y, x) for x, y in geom.exterior.coords]
                height = row.get('height', 10)
                try: height = float(height) if height else 10
                except: height = 10
                score = 85 if (area_m2 > 200 and height < 25) else (60 if area_m2 > 100 else 30)
                popup = styled_popup("مبنى", PALETTE['building'],
                    [("المساحة", f"{area_m2:.0f} م²"),
                     ("الارتفاع", f"{height:.0f} م"),
                     ("كفاءة شمسية", f"{score}%")], icon="🏢")
                folium.Polygon(coords, color=PALETTE['building_border'], weight=1.5,
                    fillColor=PALETTE['building'], fillOpacity=0.4,
                    popup=popup, tooltip=f"🏢 {area_m2:.0f} م²").add_to(fg_existing)
                if score >= 60:
                    cs = PALETTE['solar_high'] if score >= 80 else PALETTE['solar_mid']
                    dot_marker(cy, cx, cs, "☀️", f"☀️ {score}%",
                        styled_popup("سطح شمسي", cs,
                            [("الكفاءة", f"{score}%"), ("المساحة", f"{area_m2:.0f} م²")], "☀️"),
                        size=22).add_to(fg_solar)

    # خضرة
    if not gdf.empty:
        gm = pd.Series(False, index=gdf.index)
        if 'leisure' in gdf.columns:
            gm |= gdf['leisure'].isin(['park','garden','pitch']).fillna(False)
        if 'landuse' in gdf.columns:
            gm |= gdf['landuse'].isin(['forest','grass','meadow']).fillna(False)
        if 'natural' in gdf.columns:
            gm |= gdf['natural'].isin(['wood','scrub','grassland']).fillna(False)
        gz = gdf[gm]
        try: g_4326 = gz.to_crs(4326)
        except: g_4326 = gz
        for _, row in g_4326.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty: continue
            try: g_area = gpd.GeoSeries([geom], crs=4326).to_crs(3857).area.iloc[0]
            except: g_area = 0
            if geom.geom_type == 'Polygon':
                coords = [(y, x) for x, y in geom.exterior.coords]
                folium.Polygon(coords, color=PALETTE['green_border'], weight=2,
                    fillColor=PALETTE['green'], fillOpacity=0.5,
                    popup=styled_popup("منطقة خضراء", PALETTE['green'],
                        [("المساحة", f"{g_area:,.0f} م²")], "🌳"),
                    tooltip=f"🌳 {g_area:.0f} م²").add_to(fg_existing)
            try:
                c = geom.representative_point(); green_centroids.append((c.y, c.x))
            except: pass

    # مياه
    if not gdf.empty and 'natural' in gdf.columns:
        w = gdf[gdf['natural'] == 'water']
        try: w_4326 = w.to_crs(4326)
        except: w_4326 = w
        for _, row in w_4326.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty: continue
            if geom.geom_type == 'Polygon':
                coords = [(y, x) for x, y in geom.exterior.coords]
                folium.Polygon(coords, color=PALETTE['water_border'], weight=2,
                    fillColor=PALETTE['water'], fillOpacity=0.7,
                    popup=styled_popup("مسطح مائي", PALETTE['water'], [("النوع","مياه")], "💧"),
                    tooltip="💧 مياه").add_to(fg_existing)

    # مسارات
    if not gdf.empty and 'highway' in gdf.columns:
        r = gdf[gdf['highway'].notna()]
        try: r_4326 = r.to_crs(4326)
        except: r_4326 = r
        for _, row in r_4326.iterrows():
            geom = row.geometry
            if geom is None: continue
            if geom.geom_type == 'LineString':
                coords = [(y, x) for x, y in geom.coords]
                folium.PolyLine(coords, color=PALETTE['path'], weight=3,
                    opacity=0.7, popup="🛣️ مسار").add_to(fg_roads)

    # Heatmap
    if not gdf.empty and 'building' in gdf.columns:
        b = gdf[gdf['building'].notna()]
        try: b_wgs = b.to_crs(4326)
        except: b_wgs = b
        hp = []
        for p in b_wgs.geometry:
            if p is None or p.is_empty: continue
            try: c = p.representative_point(); hp.append([c.y, c.x])
            except: pass
        if hp:
            HeatMap(hp, radius=18, blur=12, min_opacity=0.35,
                gradient={0.2:'blue',0.4:'cyan',0.6:'lime',0.8:'yellow',1:'red'}).add_to(fg_heat)

    # أشجار
    n_trees = min(st.session_state.get('trees_count', 50), 100)
    tc = []
    if green_centroids:
        for gc in green_centroids:
            for _ in range(6):
                tc.append((gc[0] + np.random.uniform(-0.0004, 0.0004),
                           gc[1] + np.random.uniform(-0.0004, 0.0004)))
    if len(tc) < n_trees:
        dl = radius / 111320 * 0.7
        for _ in range(n_trees * 2):
            tc.append((center[0] + np.random.uniform(-dl, dl),
                       center[1] + np.random.uniform(-dl, dl)))
    if tc and HAS_SKLEARN and len(tc) >= n_trees:
        X = np.array(tc)
        km = KMeans(n_clusters=min(n_trees, len(X)), n_init=5, random_state=42).fit(X)
        for i, c in enumerate(km.cluster_centers_):
            dot_marker(c[0], c[1], PALETTE['tree'], "🌳", f"🌳 شجرة {i+1}",
                styled_popup(f"شجرة #{i+1}", PALETTE['tree'],
                    [("النوع","شجرة محلية"), ("التوزيع","K-Means")], "🌳"), size=20).add_to(fg_trees)

    # سلال
    waste_kg = metrics.get('waste_kg', 10)
    bins_cap = st.session_state.get('bins_capacity', 40)
    n_bins = max(1, int(np.ceil(waste_kg / bins_cap)))
    bin_types = [('عضوي', PALETTE['bin_organic'], '🥬'),
                 ('بلاستيك', PALETTE['bin_plastic'], '♳'),
                 ('ورق', PALETTE['bin_paper'], '📄'),
                 ('زجاج', PALETTE['bin_glass'], '🍶'),
                 ('عام', PALETTE['bin_general'], '🗑️')]
    bc = road_points if road_points else []
    if len(bc) < n_bins:
        dl = radius / 111320 * 0.5
        for _ in range(n_bins * 3):
            bc.append((center[0] + np.random.uniform(-dl, dl),
                       center[1] + np.random.uniform(-dl, dl)))
    if bc and HAS_SKLEARN and len(bc) >= n_bins:
        X = np.array(bc)
        km = KMeans(n_clusters=min(n_bins, len(X)), n_init=5, random_state=42).fit(X)
        for i, c in enumerate(km.cluster_centers_):
            bt = bin_types[i % 5]
            dot_marker(c[0], c[1], bt[1], bt[2], f"🗑️ {bt[0]}",
                styled_popup(f"سلة {bt[0]}", bt[1],
                    [("النوع", bt[0]), ("السعة", f"{bins_cap} كجم"),
                     ("الرقم", f"{i+1}/{n_bins}")], "🗑️"), size=22).add_to(fg_bins)

    # إنارة
    lp = road_points[::max(1, len(road_points)//25)] if road_points else []
    if not lp:
        dl = radius / 111320 * 0.7
        for _ in range(20):
            lp.append((center[0] + np.random.uniform(-dl, dl),
                       center[1] + np.random.uniform(-dl, dl)))
    for i, (lat, lon) in enumerate(lp[:30]):
        dot_marker(lat, lon, PALETTE['light'], "💡", "💡 LED",
            styled_popup(f"إنارة LED #{i+1}", PALETTE['light'],
                [("النوع","LED"), ("الكفاءة",">120 lm/W")], "💡"), size=18).add_to(fg_lights)

    # ري
    irr_targets = green_centroids[:8] if green_centroids else []
    if not irr_targets:
        dl = radius / 111320 * 0.4
        for _ in range(5):
            irr_targets.append((center[0] + np.random.uniform(-dl, dl),
                                center[1] + np.random.uniform(-dl, dl)))
    for i, (lat, lon) in enumerate(irr_targets):
        dot_marker(lat, lon, PALETTE['irrigation'], "💧", "💧 ري",
            styled_popup(f"نقطة ري #{i+1}", PALETTE['irrigation'],
                [("النوع","نقطة ري/خزان")], "💧"), size=20).add_to(fg_water)

    # ممرات حيوية
    if len(green_centroids) >= 2:
        for i in range(min(len(green_centroids)-1, 6)):
            coords = [green_centroids[i], green_centroids[i+1]]
            folium.PolyLine(coords, color=PALETTE['corridor'], weight=3,
                dash_array='10,12', opacity=0.75,
                popup="🦋 ممر حيوي").add_to(fg_corridors)

    # Marker المركز
    folium.Marker(location=list(center),
        icon=folium.DivIcon(html=f'''
            <div style="filter:drop-shadow(0 4px 12px rgba(0,0,0,0.5));">
                <div style="background:radial-gradient(circle,#ef4444 0%,#dc2626 100%);
                    width:36px;height:36px;border-radius:50%;border:3px solid white;
                    display:flex;align-items:center;justify-content:center;font-size:18px;
                    animation:pulse 2s infinite;">📍</div></div>
            <style>@keyframes pulse {{
                0%,100%{{transform:scale(1);box-shadow:0 0 0 0 rgba(239,68,68,0.7);}}
                50%{{transform:scale(1.08);box-shadow:0 0 0 12px rgba(239,68,68,0);}}}}</style>''',
            icon_size=(36,36), icon_anchor=(18,18)),
        popup=styled_popup("مركز التحليل", "#ef4444",
            [("Latitude", f"{center[0]:.5f}"), ("Longitude", f"{center[1]:.5f}"),
             ("النطاق", f"{radius} م")], "📍")).add_to(m)

    for fg in [fg_existing, fg_heat, fg_solar, fg_roads, fg_trees, fg_bins, fg_lights, fg_water, fg_corridors]:
        fg.add_to(m)

    stats = st.session_state.get('site_stats', {})
    m.get_root().html.add_child(folium.Element(build_legend(overall, status_txt, perf_color)))
    m.get_root().html.add_child(folium.Element(build_stats_panel(stats)))

    MiniMap(toggle_display=True, position='bottomright').add_to(m)
    MeasureControl(position='topright').add_to(m)
    Fullscreen(position='topright').add_to(m)
    folium.LayerControl(collapsed=False, position='topright').add_to(m)
    return m

# ============================================================
# 6. الشريط الجانبي
# ============================================================
with st.sidebar:
    st.title("🎓 Smart Planner PhD")
    st.caption("v5.0 | Phase 1+2+3 + Chat AI")

    with st.expander("📍 الموقع", expanded=True):
        q_type = st.radio("الإدخال", ["📍 موقعي الحالي", "اسم مكان", "إحداثيات"])
        q_value = ""
        if q_type == "📍 موقعي الحالي":
            loc = streamlit_geolocation()
            if loc and loc.get("latitude") and float(loc.get("latitude", 0)) != 0:
                st.session_state['my_location'] = (float(loc["latitude"]), float(loc["longitude"]))
                st.success("✅ تم تحديد موقعك")
            if st.session_state['my_location']:
                st.caption(f"📌 {st.session_state['my_location'][0]:.4f}, {st.session_state['my_location'][1]:.4f}")
        elif q_type == "اسم مكان":
            q_value = st.text_input("اسم المكان", "حديقة الأزهر، القاهرة")
        else:
            q_value = st.text_input("الإحداثيات", "30.0444, 31.2357")
        radius = st.slider("نطاق التحليل (م)", 200, 2000, 500, 100)

    with st.expander("📐 بيانات الموقع", expanded=True):
        st.session_state['site_area_m2'] = st.number_input("المساحة (م²)", min_value=100.0, value=st.session_state['site_area_m2'])
        st.session_state['soil_type'] = st.selectbox("التربة", ["Sandy", "Loamy", "Clay"],
            format_func=lambda x: {"Sandy": "رملية", "Loamy": "طميية", "Clay": "طينية"}[x])
        if st.button("✅ تأكيد البيانات"):
            st.session_state['data_quality']['soil'] = True
            st.toast("تم ✅")

    with st.expander("🌦️ المناخ"):
        st.session_state['climate_temp'] = st.number_input("الحرارة °C", value=st.session_state['climate_temp'])
        st.session_state['rainfall_mm'] = st.number_input("أمطار mm", value=st.session_state['rainfall_mm'])

    st.divider()
    st.caption("✅ Phase 1: Prototype")
    st.caption("✅ Phase 2: AI & GIS")
    st.caption("✅ Phase 3: Smart Platform")

# ============================================================
# 7. تعريف التبويبات (مهم جداً قبل الاستخدام)
# ============================================================
tabs = st.tabs(["🗺️ الخريطة الاحترافية", "💧 المياه", "⚡ الطاقة",
                "🗑️ النفايات", "📊 KPIs", "🔄 السيناريوهات", "📡 IoT", "🤖 AI Chat"])

m = compute_all_metrics()
scores = compute_scores(m)

# ============================================================
# Tab 1: الخريطة
# ============================================================
with tabs[0]:
    st.header("🗺️ الخريطة التفاعلية + التحليل المكاني")
    center = resolve_any_location(q_type, q_value)

    if center is None:
        st.warning("⚠️ لم يتم تحديد الموقع. أدخل اسم مكان صحيح أو إحداثيات.")
        st.info("💡 جرب: 'حديقة الأزهر، القاهرة' | 'الحديقة الدولية، مدينة نصر' | '30.0400, 31.2600'")
    else:
        gdf = fetch_site_data_enhanced(center, radius)
        stats = compute_actual_stats(gdf)
        st.session_state['site_stats'] = stats

        st.markdown("### 📊 بيانات مكتشفة من OpenStreetMap")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🏢 مبانٍ", f"{stats['buildings_count']:,}", help=f"مساحة: {stats['buildings_area']:,.0f} م²")
        c2.metric("🌳 مناطق خضراء", f"{stats['green_count']:,}", help=f"مساحة: {stats['green_area']:,.0f} م²")
        c3.metric("💧 مسطحات مائية", f"{stats['water_count']:,}", help=f"مساحة: {stats['water_area']:,.0f} م²")
        c4.metric("🛣️ مسارات", f"{stats['roads_count']:,}")

        st.markdown("### 🗺️ الخريطة التفاعلية")
        st.caption("💡 اضغط على أي عنصر لعرض تفاصيله | استخدم الطبقات (يمين) لإظهار/إخفاء العناصر")

        try:
            phd_map = build_phd_map(center, radius, gdf, m, scores)
            st_folium(phd_map, height=650, use_container_width=True, returned_objects=[])
        except Exception as e:
            st.error(f"خطأ في رسم الخريطة: {e}")
            st.exception(e)

        st.divider()
        st.markdown(f"### 🔍 تحليل مفصل للمنطقة")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🎯 ما تم اكتشافه")
            st.write(f"- **مبانٍ**: {stats['buildings_area']:,.0f} م²")
            st.write(f"- **خضرة**: {stats['green_area']:,.0f} م²")
            st.write(f"- **مياه**: {stats['water_area']:,.0f} م²")
            total = stats['buildings_area'] + stats['green_area']
            if total > 0:
                gp = (stats['green_area'] / total) * 100
                st.metric("نسبة الغطاء الأخضر", f"{gp:.1f}%",
                          delta="✅ جيد" if gp > 20 else "⚠️ يحتاج تحسين")
        with col2:
            st.markdown("#### 🎨 ما تم اقتراحه")
            n_bins_show = int(np.ceil(m['waste_kg'] / st.session_state['bins_capacity']))
            st.write(f"- 🌳 **{st.session_state['trees_count']}** شجرة (K-Means)")
            st.write(f"- 🗑️ **{n_bins_show}** سلة (5 أنواع)")
            st.write(f"- 💡 **30** إنارة LED")
            st.write(f"- 💧 **8** نقاط ري")
            st.write(f"- 🦋 **ممرات حيوية**")

        st.divider()
        st.markdown("#### 💡 توصيات لهذه المنطقة")
        recs = []
        if stats['green_count'] == 0:
            recs.append("🌱 لا توجد مناطق خضراء - اقتراح: تخصيص 30% كمساحات خضراء.")
        elif stats['green_area'] < 500:
            recs.append("🌱 مساحة خضراء محدودة - اقتراح: توسيعها.")
        if stats['buildings_count'] > 20:
            recs.append(f"☀️ {stats['buildings_count']} مبنى - اقتراح: ألواح شمسية.")
        if stats['roads_count'] > 5:
            recs.append(f"💡 {stats['roads_count']} مسار - اقتراح: LED + سلال K-Means.")
        for r in recs: st.info(r)

# ============================================================
# Tab 2: المياه
# ============================================================
with tabs[1]:
    st.subheader("💧 إدارة المياه")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state['irrigated_area_pct'] = st.slider("مساحة مزروعة %", 0, 100, st.session_state['irrigated_area_pct'])
        st.session_state['irrigation_system'] = st.selectbox("الري", ["Drip", "Sprinkler", "Manual"],
            format_func=lambda x: {"Drip": "تنقيط", "Sprinkler": "رش", "Manual": "يدوي"}[x])
        st.session_state['water_source'] = st.selectbox("المصدر", ["Fresh Water", "Treated Wastewater", "Rainwater"],
            format_func=lambda x: {"Fresh Water": "عذبة", "Treated Wastewater": "معالجة", "Rainwater": "أمطار"}[x])
    with c2:
        st.session_state['tree_species_count'] = st.number_input("أنواع أشجار", min_value=1, value=st.session_state['tree_species_count'])
        st.session_state['trees_count'] = st.number_input("عدد أشجار", min_value=1, value=st.session_state['trees_count'])
        if st.session_state['tree_species_count'] < 10:
            st.error("🚨 تحتاج 10 أنواع (10/20/30).")
    m = compute_all_metrics()
    c1, c2, c3 = st.columns(3)
    c1.metric("يومي (لتر)", f"{m['daily_water']:,.0f}")
    c2.metric("سنوي (م³)", f"{m['annual_m3']:,.0f}")
    c3.metric("كثافة (L/m²/day)", f"{m['water_m2']:.2f}")
    suitable = [p for p, d in NATIVE_PLANTS_DB.items() if st.session_state['soil_type'] in d['soil']]
    st.info(f"🌿 نباتات مقترحة: {' | '.join(suitable)}")

# ============================================================
# Tab 3: الطاقة
# ============================================================
with tabs[2]:
    st.subheader("⚡ الطاقة والكربون")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state['lighting_hours'] = st.slider("ساعات إنارة", 4, 14, st.session_state['lighting_hours'])
        st.session_state['solar_pct'] = st.slider("شمسية %", 0, 100, st.session_state['solar_pct'])
        st.session_state['has_pumps'] = st.checkbox("مضخات", value=st.session_state['has_pumps'])
    m = compute_all_metrics()
    c1, c2, c3 = st.columns(3)
    c1.metric("طاقة (kWh)", f"{m['total_kwh']:,.0f}")
    c2.metric("بصمة (kg CO2)", f"{m['carbon_kg']:,.0f}")
    c3.metric("صافي (kg)", f"{m['net_carbon']:,.0f}")
    if m['net_carbon'] < 0: st.success(f"🌱 Carbon Negative بـ {abs(m['net_carbon']):,.0f} kg!")

# ============================================================
# Tab 4: النفايات
# ============================================================
with tabs[3]:
    st.subheader("🗑️ النفايات والسلوك")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state['visitors_per_day'] = st.number_input("زوار", value=st.session_state['visitors_per_day'])
        st.session_state['bins_capacity'] = st.number_input("سعة سلة (كجم)", value=st.session_state['bins_capacity'])
        st.session_state['has_composting'] = st.checkbox("Composting", value=st.session_state['has_composting'])
    with c2:
        st.session_state['flow_type'] = st.selectbox("نمط الحركة", ["Linear", "Radial", "Network"],
            format_func=lambda x: {"Linear": "خطي", "Radial": "شعاعي", "Network": "شبكي"}[x])
    m = compute_all_metrics()
    bins = int(np.ceil(m['waste_kg'] / st.session_state['bins_capacity']))
    c1, c2, c3 = st.columns(3)
    c1.metric("نفايات/يوم", f"{m['waste_kg']:.1f} kg")
    c2.metric("سلال", f"{bins}")
    c3.metric("تدوير", f"{m['recycling']*100:.0f}%")

# ============================================================
# Tab 5: KPIs
# ============================================================
with tabs[4]:
    st.header("📊 مؤشرات كفاءة الموارد")
    m = compute_all_metrics(); scores = compute_scores(m)
    overall = scores['resource']
    status = "ممتاز" if overall >= 80 else "جيد" if overall >= 60 else "متوسط" if overall >= 40 else "يحتاج تحسين"

    fig = go.Figure(go.Indicator(mode="gauge+number", value=overall,
        title={'text': f"الكفاءة الشاملة: {status}"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#2E86C1"},
               'steps': [{'range': [0,40],'color':'#FADBD8'},{'range': [40,60],'color':'#FDEBD0'},
                         {'range': [60,80],'color':'#FCF3CF'},{'range': [80,100],'color':'#D5F5E3'}]}))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    cats = ['مياه','طاقة','نفايات','كربون','تنوع','موارد']
    vals = [scores['water'], scores['energy'], scores['waste'], scores['carbon'], scores['biodiversity'], scores['resource']]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=cats+[cats[0]], fill='toself', name='الحالي', line_color='#27AE60'))
    fig.add_trace(go.Scatterpolar(r=[100]*7, theta=cats+[cats[0]], fill='toself', name='المستهدف',
        line_color='#2E86C1', line_dash='dash', fillcolor='rgba(46,134,193,0.05)'))
    fig.update_layout(height=450, polar=dict(radialaxis=dict(range=[0, 100])))
    st.plotly_chart(fig, use_container_width=True)

    gd = [('💧 المياه', scores['water'], '#2E86C1'), ('⚡ الطاقة', scores['energy'], '#F39C12'),
          ('♻️ النفايات', scores['waste'], '#27AE60'), ('🌍 الكربون', scores['carbon'], '#16A085'),
          ('🌳 التنوع', scores['biodiversity'], '#8E44AD'), ('📊 الموارد', scores['resource'], '#E74C3C')]
    cols = st.columns(3)
    for i, (name, val, color) in enumerate(gd):
        with cols[i % 3]:
            fg = go.Figure(go.Indicator(mode="gauge+number", value=val,
                title={'text': name, 'font': {'size': 14}},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color},
                       'steps': [{'range': [0,40],'color':'#FADBD8'},{'range': [40,60],'color':'#FDEBD0'},
                                 {'range': [60,80],'color':'#FCF3CF'},{'range': [80,100],'color':'#D5F5E3'}]}))
            fg.update_layout(height=200, margin=dict(t=50, b=10, l=20, r=20))
            st.plotly_chart(fg, use_container_width=True)

# ============================================================
# Tab 6: السيناريوهات
# ============================================================
with tabs[5]:
    st.header("🔄 محاكاة السيناريوهات")
    scenarios = {
        "A - توفير المياه": compute_all_metrics({'irr_sys': 'Drip', 'irr_pct': max(30, st.session_state['irrigated_area_pct']-10), 'water_src': 'Treated Wastewater', 'compost': True}),
        "B - التوازن": compute_all_metrics({'solar': min(100, st.session_state['solar_pct']+20), 'compost': True, 'species': max(10, st.session_state['tree_species_count'])}),
        "C - التنوع الأقصى": compute_all_metrics({'species': 20, 'trees': max(200, st.session_state['trees_count']), 'solar': 60, 'compost': True, 'water_src': 'Treated Wastewater'})
    }
    rows = []
    for name, sm in scenarios.items():
        sc = compute_scores(sm)
        rows.append({'السيناريو': name, 'مياه': sc['water'], 'طاقة': sc['energy'],
                     'نفايات': sc['waste'], 'كربون': sc['carbon'], 'تنوع': sc['biodiversity'], 'إجمالي': sc['resource']})
    df_sc = pd.DataFrame(rows)
    fig = go.Figure()
    for p in ['مياه','طاقة','نفايات','كربون','تنوع']:
        fig.add_trace(go.Bar(name=p, x=df_sc['السيناريو'], y=df_sc[p]))
    fig.update_layout(barmode='group', height=400, yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_sc, use_container_width=True, hide_index=True)
    best = df_sc.loc[df_sc['إجمالي'].idxmax(), 'السيناريو']
    st.success(f"🏆 **السيناريو الموصى به**: {best}")

# ============================================================
# Tab 7: IoT
# ============================================================
with tabs[6]:
    st.header("📡 محاكاة IoT")
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("▶️ بدء"):
            st.session_state['iot_running'] = True
        if st.button("⏹️ إيقاف"):
            st.session_state['iot_running'] = False
    if st.session_state['iot_running']:
        now = datetime.now(); n = 20
        df_iot = pd.DataFrame({
            'time': [now - timedelta(minutes=i*5) for i in range(n)][::-1],
            'soil': [random.uniform(25, 65) for _ in range(n)],
            'tank': [random.uniform(40, 95) for _ in range(n)],
            'energy': [random.uniform(2, 12) for _ in range(n)],
            'temp': [random.uniform(22, 38) for _ in range(n)]})
        latest = df_iot.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💧 رطوبة", f"{latest['soil']:.1f}%")
        c2.metric("🚰 خزان", f"{latest['tank']:.1f}%")
        c3.metric("⚡ kW", f"{latest['energy']:.1f}")
        c4.metric("🌡️ حرارة", f"{latest['temp']:.1f}°C")
        for col, label in [('soil','رطوبة %'), ('tank','خزان %'), ('energy','استهلاك kW'), ('temp','حرارة °C')]:
            fig = px.line(df_iot, x='time', y=col, title=label, markers=True)
            fig.update_layout(height=230)
            st.plotly_chart(fig, use_container_width=True)
        if latest['tank'] < 30: st.error("🚨 خزان منخفض!")
        if latest['soil'] < 25: st.warning("⚠️ يحتاج ري.")
    else:
        st.info("👈 اضغط 'بدء'.")

# ============================================================
# Tab 8: AI Chat Assistant
# ============================================================
with tabs[7]:
    st.header("🤖 المساعد الذكي — محادثة تفاعلية")
    st.caption("اسألني عن أي شيء: مياه، طاقة، نفايات، تنوع، حسابات، توصيات، سيناريوهات...")

    # ---- دوال الإجابة ----
    def classify_intent(question):
        q = question.lower(); intents = set()
        kws = {
            'water': ['مياه','ماء','ري','تربة','سقي','water','irrigation'],
            'energy': ['طاقة','كهرباء','إنارة','انارة','شمسية','مضخات','led','energy','solar'],
            'waste': ['نفايات','نفاية','سلة','سلال','تدوير','compost','سماد','waste','recycling'],
            'carbon': ['كربون','انبعاث','بصمة','carbon','emission','co2'],
            'biodiversity': ['تنوع','أشجار','اشجار','نباتات','أنواع','انواع','biodiversity','trees'],
            'human': ['زوار','سلوك','حركة','مسار','behavior','visitors','flow'],
            'climate': ['مناخ','حرارة','أمطار','امطار','رياح','climate','temperature'],
            'soil': ['تربة','رملية','طينية','طميية','soil'],
            'kpi': ['kpi','مؤشر','مؤشرات','أداء','اداء','تقييم','benchmark'],
            'scenario': ['سيناريو','بديل','بدائل','scenario'],
            'summary': ['ملخص','خلاصة','تقرير','summary','report','كل حاجة'],
        }
        for intent, kw in kws.items():
            if any(k in q for k in kw): intents.add(intent)
        return intents if intents else {'general'}

    def gen_response(q, m, s, api_key=None):
        intents = classify_intent(q)

        if api_key and HAS_OPENAI:
            try:
                client = OpenAI(api_key=api_key)
                ctx = f"""أنت خبير هندسي بيئي (PhD). بيانات الموقع:
- المساحة: {m['area']} م² | التربة: {st.session_state['soil_type']} | الزوار: {m['visitors']}
- المياه: {m['water_m2']:.2f} L/m²/day (كفاءة {s['water']}%) | Benchmark: 12 (Shammas 1991)
- الطاقة: {m['energy_m2']:.2f} kWh/m²/yr (كفاءة {s['energy']}%) | Benchmark: 3-7 (BBP 2025)
- النفايات: {m['waste_kg']:.2f} kg/day | تدوير {m['recycling']*100:.0f}% (كفاءة {s['waste']}%)
- الكربون: Net {m['net_carbon']:.0f} kg (كفاءة {s['carbon']}%)
- التنوع: {m['species']} أنواع (كفاءة {s['biodiversity']}%) | Target 10 (Santamour 1990)
- الكفاءة الشاملة: {s['resource']}%

أجب بالعربية الفصحى الأكاديمية مع أرقام، حسابات، ومراجع من PDF."""
                resp = client.chat.completions.create(model="gpt-4o-mini",
                    messages=[{"role": "system", "content": ctx}, {"role": "user", "content": q}],
                    temperature=0.4, max_tokens=1200)
                return resp.choices[0].message.content
            except Exception as e:
                st.warning(f"⚠️ تعذر الاتصال بـ AI ({e}). سأستخدم النظام الخبير.")

        parts = []
        if 'water' in intents:
            gap = m['water_m2'] - 12
            parts.append(f"""### 💧 تحليل المياه
| المؤشر | القيمة |
|---|---|
| الاستهلاك اليومي | {m['daily_water']:,.0f} لتر |
| الاستهلاك السنوي | {m['annual_m3']:,.0f} م³ |
| الكثافة | {m['water_m2']:.2f} L/m²/day |
| **Benchmark** | **12 L/m²/day** |
| كفاءة الري | {s['water']}% |
| المصدر | {m['water_src']} |
| النظام | {m['irr_sys']} |

**📚 المرجع:** Shammas 1991 — Riyadh Parks

**📌 الحالة:** {'✅ مطابق' if gap <= 0 else f'⚠️ يتجاوز بـ {gap:.2f}'}

**💡 التوصيات:**
- {'استمر، أنت أفضل من المعيار!' if gap <= 0 else 'خفّض الاستهلاك عبر تحسين الري'}
- {'بدّل للتنقيط (Drip) لتوفير ~15%' if m['irr_sys'] != 'Drip' else '✅ نظام التنقيط ممتاز (90%)'}
- {'استخدم مياه معالجة (EPA 2012)' if m['water_src'] == 'Fresh Water' else '✅ استخدام المياه المعالجة مطابق'}
- {'أضف Compost للتربة الرملية (USDA)' if st.session_state['soil_type'] == 'Sandy' else ''}""")

        if 'energy' in intents:
            parts.append(f"""### ⚡ تحليل الطاقة
| المؤشر | القيمة |
|---|---|
| إجمالي الطاقة | {m['total_kwh']:,.0f} kWh/سنة |
| الكثافة | {m['energy_m2']:.2f} kWh/m²/yr |
| **Benchmark** | **3 (جيد) - 7 (نموذجي)** |
| التغطية الشمسية | {m['solar']}% |
| الكفاءة | {s['energy']}% |

**📚 المرجع:** BBP 2025 — REEB

**💡 التوصيات:**
- {'☀️ ارفع الشمسية إلى 50%+ لخفض ' + f"{(m['total_kwh']*(50-m['solar'])/100):,.0f} kWh" if m['solar'] < 50 else '✅ التغطية الشمسية ممتازة'}
- 💡 LED بدل الهالوجين (توفير 50-70%)
- {'⚡ قلل ساعات الإنارة من ' + str(st.session_state['lighting_hours']) + ' إلى 8' if st.session_state['lighting_hours'] > 10 else '✅ ساعات الإنارة مناسبة'}""")

        if 'waste' in intents:
            bins = int(np.ceil(m['waste_kg'] / st.session_state['bins_capacity']))
            parts.append(f"""### 🗑️ تحليل النفايات
| المؤشر | القيمة |
|---|---|
| نفايات الزوار/يوم | {m['waste_kg']:.2f} kg |
| النفايات الخضراء/سنة | {m['irr_area']*1.75:,.0f} kg |
| معدل التدوير | {m['recycling']*100:.0f}% |
| **المستهدف** | **70%** (Cooley 2018) |
| السلال المطلوبة | {bins} |
| Composting | {'✅' if m['compost'] else '❌'} |

**📚 المراجع:** Mukhtar 2016 + Song 2025 + Cooley 2018

**💡 التوصيات:**
- {'♻️ فعّل Composting لرفع التدوير إلى 35%' if not m['compost'] else '✅ Composting مفعل'}
- 🗑️ 5 أنواع سلال: عضوي، بلاستيك، ورق، زجاج، عام
- 📍 توزيع وفق حركة الزوار (Linear/Radial/Network)
- ⚠️ تجنب تلوث التدوير فوق 30%""")

        if 'carbon' in intents:
            st_txt = "🌱 Carbon Negative" if m['net_carbon'] < 0 else "🔴 Carbon Positive"
            parts.append(f"""### 🌍 تحليل الكربون
| المؤشر | القيمة |
|---|---|
| انبعاثات الشبكة | {m['carbon_kg']:,.0f} kg CO2e |
| الامتصاص | {m['carbon_seq']:,.0f} kg CO2e |
| **الصافي** | **{m['net_carbon']:,.0f} kg** |
| الكفاءة | {s['carbon']}% |

**📚 المرجع:** Zhou 2025 + Basaly 2019

**📌 الحالة:** {st_txt}

**💡 التوصيات:**
{'- 🌳 ازرع أشجار إضافية للتعويض' if m['net_carbon'] > 0 else '- ✅ ممتاز، حافظ على الأداء'}
- 🚗 مركبات كهربائية للصيانة
- 🏗️ مواد منخفضة الكربون (Guo 2023)""")

        if 'biodiversity' in intents:
            gap = 10 - m['species']
            parts.append(f"""### 🌳 تحليل التنوع البيولوجي
| المؤشر | القيمة |
|---|---|
| أنواع الأشجار | {m['species']} |
| **المستهدف** | **10** (قاعدة 10/20/30) |
| إجمالي الأشجار | {m['trees']} |
| كثافة الأشجار | {(m['trees']/(m['area']/10000)):.0f} شجرة/هكتار |
| **المستهدف** | **150** (Lindén 2020) |

**📚 المراجع:** Santamour 1990 + Lindén 2020 + DeepRoot

**💡 التوصيات:**
{'- 🌱 أضف ' + str(gap) + ' أنواع للوصول لقاعدة 10/20/30' if gap > 0 else '✅ التنوع مطابق'}
- 🌾 استخدم الأنواع المحلية فقط
- 🌱 حجم التربة: 17/28.3/42.5 م³
- 🦋 ممرات حيوية بين المناطق الخضراء""")

        if 'human' in intents:
            flow = st.session_state.get('flow_type', 'Linear')
            parts.append(f"""### 🚶‍♂️ السلوك البشري
| المؤشر | القيمة |
|---|---|
| الزوار/يوم | {m['visitors']} |
| نفايات/زائر | 0.05 kg |
| نمط الحركة | {flow} |

**📚 المرجع:** دراسة السلوك البشري - PDF

**💡 التوصيات حسب النمط:**
- **Linear**: توزيع السلال على طول المسار
- **Radial**: حول النقطة المركزية
- **Network**: عند تقاطعات الشبكة
- ♿ مسارات ميسّرة لكبار السن""")

        if 'climate' in intents:
            parts.append(f"""### 🌦️ المناخ
| المؤشر | القيمة |
|---|---|
| الحرارة | {st.session_state['climate_temp']}°C |
| الأمطار | {st.session_state['rainfall_mm']} mm |

**📚 المرجع:** PDF - North Orientation

**💡 التوصيات:**
- 🧭 اتجاه الشمال مهم للظل
- ☀️ حركة الشمس تؤثر على الطاقة الشمسية
- 🌧️ الأمطار تقلل احتياج الري""")

        if 'soil' in intents:
            soil_info = {
                'Sandy': ('< 1.6 g/cm³', 'منخفضة', 'أضف Compost و Mulch'),
                'Loamy': ('< 1.4 g/cm³', 'متوسطة', 'تربة مثالية'),
                'Clay': ('< 1.4 g/cm³', 'عالية', 'أضف رمل لتحسين التصريف'),
            }[st.session_state['soil_type']]
            parts.append(f"""### 🌱 التربة: {st.session_state['soil_type']}
| المؤشر | القيمة |
|---|---|
| الكثافة | {soil_info[0]} |
| الاحتفاظ | {soil_info[1]} |
| الإجراء | {soil_info[2]} |

**📚 المرجع:** USDA NRCS 2023 + FHWA 2015""")

        if 'kpi' in intents:
            parts.append(f"""### 📊 مؤشرات الأداء
| المؤشر | الحالي | Baseline | Target | كفاءة |
|---|---|---|---|---|
| 💧 المياه | {m['water_m2']:.2f} | 12 | 8 | {s['water']}% |
| ⚡ الطاقة | {m['energy_m2']:.2f} | 7 | 3 | {s['energy']}% |
| ♻️ التدوير | {m['recycling']*100:.0f}% | 10 | 70 | {s['waste']}% |
| 🌍 الكربون | {m['net_carbon']:,.0f} | {m['carbon_kg']:,.0f} | 0 | {s['carbon']}% |
| 🌳 التنوع | {m['species']} | 5 | 10 | {s['biodiversity']}% |
| **📊 الإجمالي** | - | - | - | **{s['resource']}%** |""")

        if 'scenario' in intents:
            sc_data = {
                "A": compute_all_metrics({'irr_sys': 'Drip', 'water_src': 'Treated Wastewater', 'compost': True}),
                "B": compute_all_metrics({'solar': min(100, st.session_state['solar_pct']+20), 'compost': True, 'species': max(10, st.session_state['tree_species_count'])}),
                "C": compute_all_metrics({'species': 20, 'trees': max(200, st.session_state['trees_count']), 'solar': 60, 'compost': True})
            }
            rows_txt = ""
            for name, sm in sc_data.items():
                sc = compute_scores(sm)
                rows_txt += f"| {name} | {sc['water']}% | {sc['energy']}% | {sc['waste']}% | {sc['carbon']}% | {sc['biodiversity']}% | **{sc['resource']}%** |\n"
            parts.append(f"""### 🔄 مقارنة السيناريوهات
| س | مياه | طاقة | نفايات | كربون | تنوع | الإجمالي |
|---|---|---|---|---|---|---|
{rows_txt}

**🏆 الأفضل:** {max(sc_data.items(), key=lambda x: compute_scores(x[1])['resource'])[0]}""")

        if 'summary' in intents:
            parts.append(f"""### 📋 ملخص شامل
**🏗️ الموقع:** {m['area']:,.0f} م² | التربة: {st.session_state['soil_type']} | الزوار: {m['visitors']}

**📊 الكفاءات:**
- 💧 المياه: {s['water']}% ({m['water_m2']:.2f} L/m²/day)
- ⚡ الطاقة: {s['energy']}% ({m['energy_m2']:.2f} kWh/m²/yr)
- ♻️ النفايات: {s['waste']}%
- 🌍 الكربون: {s['carbon']}%
- 🌳 التنوع: {s['biodiversity']}%
- **📊 الإجمالي: {s['resource']}%**

**🎯 أولويات التحسين:**
1. {min([(s['water'],'المياه'),(s['energy'],'الطاقة'),(s['waste'],'النفايات'),(s['carbon'],'الكربون'),(s['biodiversity'],'التنوع')])[1]} - {min([s['water'],s['energy'],s['waste'],s['carbon'],s['biodiversity']])}%""")

        if not parts:
            parts.append(f"""### 👋 أهلاً! أنا المساعد الذكي
أقدر أجاوبك عن:

- 💧 **المياه** - "ما وضع المياه؟"
- ⚡ **الطاقة** - "كيف الطاقة؟"
- 🗑️ **النفايات** - "ما وضع النفايات؟"
- 🌍 **الكربون** - "ما البصمة؟"
- 🌳 **التنوع** - "التنوع البيولوجي؟"
- 📊 **KPIs** - "ملخص المؤشرات"
- 🔄 **السيناريوهات** - "قارن A,B,C"
- 📋 **الملخص** - "ملخص شامل"

**جرّب أي سؤال!**""")

        return "\n\n---\n\n".join(parts)

    # ---- واجهة الشات ----
    col_a, col_b = st.columns([3, 1])
    with col_a:
        api_key = st.text_input("🔑 OpenAI API Key (اختياري)", type="password", key="chat_api")
    with col_b:
        if st.button("🗑️ مسح", use_container_width=True):
            st.session_state.chat_history = None
            st.rerun()

    if st.session_state.chat_history is None:
        m = compute_all_metrics(); s = compute_scores(m)
        st.session_state.chat_history = [{"role": "assistant", "content": gen_response("مرحبا", m, s)}]

    m = compute_all_metrics(); s = compute_scores(m)

    st.markdown("#### 💡 أسئلة سريعة:")
    quick = [
        ("💧 المياه","ما وضع المياه؟"), ("⚡ الطاقة","كيف الطاقة؟"),
        ("🗑️ النفايات","ما وضع النفايات؟"), ("🌍 الكربون","ما البصمة الكربونية؟"),
        ("🌳 التنوع","كيف التنوع البيولوجي؟"), ("🌱 التربة","توصيات التربة؟"),
        ("📊 KPIs","ملخص KPIs"), ("🔄 السيناريوهات","قارن السيناريوهات"),
        ("📋 ملخص","ملخص شامل"), ("👥 السلوك","كيف حركة الزوار؟"),
        ("🌦️ المناخ","تأثير المناخ؟"), ("💡 توصيات","كل التوصيات"),
    ]
    for i in range(0, len(quick), 4):
        cols = st.columns(4)
        for j, (lbl, q) in enumerate(quick[i:i+4]):
            with cols[j]:
                if st.button(lbl, use_container_width=True, key=f"q_{i}_{j}"):
                    st.session_state.chat_history.append({"role": "user", "content": q})
                    with st.spinner("🤔"):
                        st.session_state.chat_history.append({"role": "assistant", "content": gen_response(q, m, s, api_key)})
                    st.rerun()

    st.divider()
    st.markdown("#### 💬 المحادثة:")
    for msg in st.session_state.chat_history:
        if msg['role'] == 'user':
            with st.chat_message("user", avatar="👤"): st.markdown(msg['content'])
        else:
            with st.chat_message("assistant", avatar="🤖"): st.markdown(msg['content'])

    if prompt := st.chat_input("اكتب سؤالك..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.spinner("🤔"):
            resp = gen_response(prompt, m, s, api_key)
        st.session_state.chat_history.append({"role": "assistant", "content": resp})
        st.rerun()

    st.divider()
    if st.button("📥 تصدير المحادثة (HTML/PDF)", use_container_width=True):
        chat_html = ""
        for msg in st.session_state.chat_history:
            role = "👤 أنت" if msg['role'] == 'user' else "🤖 المساعد"
            clr = "#3b82f6" if msg['role'] == 'user' else "#10b981"
            bg = "#eff6ff" if msg['role'] == 'user' else "#f0fdf4"
            chat_html += f'''<div style="margin:15px 0;padding:15px;background:{bg};border-radius:12px;border-right:4px solid {clr};">
                <div style="font-weight:700;color:{clr};margin-bottom:8px;">{role}</div>
                <div style="color:#1e293b;line-height:1.8;">{msg['content'].replace(chr(10), "<br>")}</div></div>'''
        full = f"""<html dir="rtl"><head><meta charset="utf-8"><style>
        body{{font-family:'Segoe UI',Tahoma;margin:40px;background:#f8fafc;color:#1e293b;}}
        h1{{color:#2E86C1;border-bottom:3px solid;padding-bottom:12px;}}</style></head><body>
        <h1>🎓 تقرير المساعد الذكي</h1>
        <p><strong>التاريخ:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        <p><strong>الكفاءة الشاملة:</strong> {s['resource']}%</p>
        {chat_html}
        </body></html>"""
        st.download_button("⬇️ تحميل", data=full,
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
            mime="text/html", type="primary", use_container_width=True)