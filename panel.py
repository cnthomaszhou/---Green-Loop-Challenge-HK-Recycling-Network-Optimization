# panel.py - 最終完整版（全中英並列 + 地圖即時切換生效 + 性能優化 + 關鍵修正 + 新增回收站点 + 气动规划）
# 最新修改：
# - 氣動點數量提升至 500 個
# - 氣動點僅針對所有有私宅的區域（不再只限未覆蓋區，也不再區分密集/稀疏）
# - 在覆蓋率已高的私宅區域也會適當放置氣動點（全域私宅 K=500 聚類）
# - 高級回收站圖例改為藍色星星，並整合到氣動規劃顯示控制區
# - 圖例與說明同步更新
# - 中英文切換時強制全英文或全中文（不再混排）
# 作者：THZH Program - Zhou Ziyue（後續修改整合）
# 運行指令: streamlit run panel.py

import streamlit as st
import pandas as pd
import numpy as np
from streamlit_folium import folium_static
import folium
from folium.plugins import FastMarkerCluster, HeatMap
import json
import geopandas as gpd
from scipy.spatial.distance import cdist
import re
import os
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import plotly.express as px

# ====================== 頁面配置 / Page Config ======================
st.set_page_config(
    page_title="香港回收網絡優化 - Green Loop Challenge / HK Recycling Network Optimization",
    page_icon="♻️",
    layout="wide"
)

# ====================== 語言切換 / Language Switch ======================
with st.sidebar:
    lang = st.selectbox("語言 / Language", ["中文 / Chinese", "English"], index=0)

is_english = (lang == "English")

# ====================== 文字字典 / Text Dictionary ======================
texts = {
    "中文 / Chinese": {
        "title": "♻️ 香港回收網絡優化 - Green Loop Challenge / HK Recycling Network Optimization",
        "loading": "正在載入資料... / Loading data...",
        "filters": "🔍 篩選條件（即時更新覆蓋率） / Filter Conditions (Live Update)",
        "district": "地區 / District",
        "waste_type": "回收類型 / Waste Type",
        "station_type": "回收點類型 / Station Type",
        "premium_only": "僅優質站點 / Premium Stations Only",
        "distance": "覆蓋距離 (米) / Coverage Distance (meters)",
        "show_public": "顯示公屋標記（紅色） / Show Public Estates Markers (Red)",
        "show_private_heat": "顯示私樓密度熱力圖 / Show Private Buildings Density Heatmap",
        "private_title": "選定地區覆蓋分析（含最近回收站資訊） / Selected District Coverage Analysis (with Nearest Station Info)",
        "covered_buildings": "符合條件的建築數 / Buildings Covered",
        "coverage_rate": "覆蓋率 / Coverage Rate",
        "of_total": "佔總數 / of Total",
        "premium_ratio": "最近為 Recycling Stations（優質站）比例 / Nearest are Recycling Stations (Premium)",
        "common_waste": "垃圾回收種類比例（選定地區） / Waste Type Proportions (Selected District)",
        "formula_title": "覆蓋率計算公式（選定地區） / Coverage Rate Formula (Selected District)",
        "recycling_points": "回收點 / Recycling Points",
        "public_coverage": "公屋覆蓋率 / Public Estates Coverage",
        "private_coverage": "私樓覆蓋率 / Private Buildings Coverage",
        "map_title": "全香港回收網絡地圖 / Hong Kong Recycling Network Map",
        "proposal": "數據驅動提案 / Data-Driven Proposal",
        "insight": "核心洞察 / Key Insight",
        "suggestion": "建議 / Recommendation",
        "impact": "預期影響 / Expected Impact",
        "no_data": "無符合條件資料 / No matching data",
        "distance_coverage_trend": "覆蓋率隨距離變化趨勢 / Coverage Rate vs. Distance Trend",
        "new_sites_title": "新增回收站点建议 / New Recycling Sites Proposal",
        "new_sites_k": "建议站点数量 (K) / Number of Proposed Sites (K)",
        "new_sites_impact": "预计覆盖提升 / Expected Coverage Improvement",
        "pneumatic_title": "气动系统规划（基于巴塞罗那方案） / Pneumatic System Planning (Barcelona-Inspired)",
        "pneumatic_formula": "规划公式 / Planning Formula",
        "pneumatic_k": "气动点数量 (固定500) / Number of Pneumatic Points (Fixed 500)",
        "pneumatic_impact": "预计覆盖提升 / Expected Coverage Improvement",
        "show_new_sites": "显示新增回收站层 / Show New Recycling Sites Layer",
        "show_pneumatic": "显示气动规划层 / Show Pneumatic Planning Layer",
        "legend_new_sites": "图例：绿色星星 = 新回收站（Recycle Station） / Legend: Green Stars = New Recycle Stations",
        "legend_pneumatic": "图例：绿色中空圆圈 = 气动点；紫色线 = 连接最近高级回收站 / Legend: Green Hollow Circles = Pneumatic Points; Purple Lines = to Nearest Premium",
        "legend_premium": "图例：蓝色星星 = 高级回收站（Premium Stations） / Legend: Blue Stars = Premium Recycling Stations"
    },
    "English": {
        "title": "♻️ Hong Kong Recycling Network Optimization - Green Loop Challenge",
        "loading": "Loading data...",
        "filters": "🔍 Filter Conditions (Live Update)",
        "district": "District",
        "waste_type": "Waste Type",
        "station_type": "Station Type",
        "premium_only": "Premium Stations Only",
        "distance": "Coverage Distance (meters)",
        "show_public": "Show Public Estates Markers (Red)",
        "show_private_heat": "Show Private Buildings Density Heatmap",
        "private_title": "Selected District Coverage Analysis (with Nearest Station Info)",
        "covered_buildings": "Buildings Covered",
        "coverage_rate": "Coverage Rate",
        "of_total": "of Total",
        "premium_ratio": "Nearest are Recycling Stations (Premium)",
        "common_waste": "Waste Type Proportions (Selected District)",
        "formula_title": "Coverage Rate Formula (Selected District)",
        "recycling_points": "Recycling Points",
        "public_coverage": "Public Estates Coverage",
        "private_coverage": "Private Buildings Coverage",
        "map_title": "Hong Kong Recycling Network Map",
        "proposal": "Data-Driven Proposal",
        "insight": "Key Insight",
        "suggestion": "Recommendation",
        "impact": "Expected Impact",
        "no_data": "No matching data",
        "distance_coverage_trend": "Coverage Rate vs. Distance Trend",
        "new_sites_title": "New Recycling Sites Proposal",
        "new_sites_k": "Number of Proposed Sites (K)",
        "new_sites_impact": "Expected Coverage Improvement",
        "pneumatic_title": "Pneumatic System Planning (Barcelona-Inspired)",
        "pneumatic_formula": "Planning Formula",
        "pneumatic_k": "Number of Pneumatic Points (Fixed 500)",
        "pneumatic_impact": "Expected Coverage Improvement",
        "show_new_sites": "Show New Recycling Sites Layer",
        "show_pneumatic": "Show Pneumatic Planning Layer",
        "legend_new_sites": "Legend: Green Stars = New Recycle Stations",
        "legend_pneumatic": "Legend: Green Hollow Circles = Pneumatic Points; Purple Lines = to Nearest Premium",
        "legend_premium": "Legend: Blue Stars = Premium Recycling Stations"
    }
}

t = texts[lang]

# ====================== 禁用滑鼠滾輪影響 / Disable Mouse Wheel Impact ======================
st.markdown("""
    <style>
        .stApp { overflow-x: hidden; }
        .stPlotlyChart, .element-container [data-testid="stPlotlyChart"], 
        [data-testid="stDeckGlJsonChart"], [kind="line_chart"], [kind="bar_chart"] {
            pointer-events: auto !important;
        }
    </style>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const iframes = document.querySelectorAll('iframe[src*="folium"]');
            iframes.forEach(iframe => {
                iframe.addEventListener('wheel', function(e) { e.stopPropagation(); e.preventDefault(); }, { passive: false });
            });
            const charts = document.querySelectorAll('[data-testid="stPlotlyChart"], .stPlotlyChart, [kind="line_chart"], [kind="bar_chart"]');
            charts.forEach(chart => {
                chart.addEventListener('wheel', function(e) { e.stopPropagation(); e.preventDefault(); }, { passive: false });
            });
            const observer = new MutationObserver(() => {
                const newCharts = document.querySelectorAll('[data-testid="stPlotlyChart"], .stPlotlyChart');
                newCharts.forEach(chart => {
                    if (!chart.dataset.wheelDisabled) {
                        chart.addEventListener('wheel', function(e) { e.stopPropagation(); e.preventDefault(); }, { passive: false });
                        chart.dataset.wheelDisabled = 'true';
                    }
                });
            });
            observer.observe(document.body, { childList: true, subtree: true });
        });
    </script>
""", unsafe_allow_html=True)

# ====================== 地區名稱標準化 / Normalize District Names ======================
def normalize_district_name(district):
    if pd.isna(district) or str(district).strip() == "":
        return "Unknown"
    d = str(district).strip().replace("_", " ").replace("And", "&").replace(" and ", " & ")
    mapping = {
        "Central Western": "Central & Western",
        "Central and Western": "Central & Western",
        "Central_Western": "Central & Western",
        "Kwai Tsing": "Kwai Tsing",
        "Yau Tsim Mong": "Yau Tsim Mong",
        "Kwun Tong": "Kwun Tong",
        "Yuen Long": "Yuen Long",
        "Tuen Mun": "Tuen Mun",
        "Sai Kung": "Sai Kung",
        "Sha Tin": "Sha Tin",
        "Tai Po": "Tai Po",
        "North": "North",
        "Islands": "Islands",
        "Eastern": "Eastern",
        "Southern": "Southern",
        "Wan Chai": "Wan Chai",
        "Sham Shui Po": "Sham Shui Po",
        "Kowloon City": "Kowloon City",
        "Wong Tai Sin": "Wong Tai Sin",
        "Tsuen Wan": "Tsuen Wan"
    }
    return mapping.get(d, d)

# ====================== 數據加載 / Data Loading ======================
@st.cache_data
def load_recycling_points():
    try:
        df = pd.read_csv("Recyclable-Collection-Points-Data.csv", encoding='utf-8')
        df.columns = df.columns.str.strip()
        mapping = {'cp_id': 'id', 'cp_state': 'status', 'district_id': 'district',
                   'address_en': 'address_en', 'address_tc': 'address_tc',
                   'lat': 'latitude', 'lgt': 'longitude', 'waste_type': 'waste_types',
                   'legend': 'type', 'openhour_tc': 'hours_tc'}
        df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
        df = df.dropna(subset=['latitude', 'longitude'])
        df = df[(df['latitude'] != 0) & (df['longitude'] != 0)]
        df = df[(df['latitude'].between(22.15, 22.55)) & (df['longitude'].between(113.8, 114.4))]
        df['waste_types'] = df['waste_types'].fillna('Unknown')
        df['type'] = df.get('type', '回收站 / Recycle Station').fillna('回收站 / Recycle Station')
        df['district'] = df.get('district', 'Unknown').fillna('Unknown')
        df['is_premium'] = df['type'].str.contains('GREEN@COMMUNITY|Recycling Station|Recycling Store', case=False, na=False)
        
        df['district'] = df['district'].apply(normalize_district_name)
        return df
    except Exception as e:
        st.error(f"回收點 CSV 加載失敗 / Recycling Points CSV Load Failed: {e}" if not is_english else f"Failed to load Recycling Points CSV: {e}")
        return pd.DataFrame()

@st.cache_data
def load_public_housing():
    path = "prh-estates.json"
    if not os.path.exists(path): return pd.DataFrame()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        estates = []
        for item in data if isinstance(data, list) else [data]:
            estate_name = item.get("Estate Name", {}).get("en", "Unknown")
            district = item.get("District Name", {}).get("en", "Unknown")
            lat = item.get("Estate Map Latitude")
            lon = item.get("Estate Map Longitude")
            if lat is None or lon is None: continue
            flats = 0
            flats_str = str(item.get("No. of Units", item.get("No. of Rental Flats", "0")))
            match = re.search(r'\d[\d\s,]*', flats_str)
            if match: flats = int(match.group().replace(" ", "").replace(",", ""))
            estates.append({'estate_name': estate_name, 'district': district, 'flats': flats,
                            'latitude': lat, 'longitude': lon})
        df = pd.DataFrame(estates)
        df['district'] = df['district'].apply(normalize_district_name)
        df['weight'] = df['flats'].fillna(0)
        return df
    except:
        return pd.DataFrame()

@st.cache_data
def load_private_buildings():
    path = "PrivateBuildings.csv"
    if not os.path.exists(path): 
        st.error(f"私宅文件不存在 / Private Buildings file not found: {path}" if not is_english else f"Private Buildings file not found: {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, usecols=['LATITUDE', 'LONGITUDE', 'SEARCH1_E'], 
                         low_memory=False, encoding='utf-8-sig', on_bad_lines='skip')
        df.columns = df.columns.str.strip()
        df = df.rename(columns={'SEARCH1_E': 'district', 'LATITUDE': 'latitude', 'LONGITUDE': 'longitude'})
        df = df.dropna(subset=['latitude', 'longitude'])
        df = df[(df['latitude'].between(22.0, 22.6)) & (df['longitude'].between(113.8, 114.5))]
        
        df['type'] = 'Private Building / Private Building'
        df['district'] = df['district'].apply(normalize_district_name)
        df['weight'] = 50
        return df[['latitude', 'longitude', 'district', 'type', 'weight']]
    except Exception as e:
        st.error(f"私宅載入失敗 / Private buildings load failed: {str(e)}" if not is_english else f"Private buildings load failed: {str(e)}")
        return pd.DataFrame()

# ====================== 覆蓋率計算 / Coverage Calculation ======================
@st.cache_data
def calculate_district_coverage(recycle_df, estate_df, distance_km=0.5):
    if recycle_df.empty or estate_df.empty:
        return {'coverage_rate': 0.0, 'covered': 0, 'total': 0}
    coords_rec = recycle_df[['latitude', 'longitude']].values
    coords_est = estate_df[['latitude', 'longitude']].values
    distances = cdist(coords_est, coords_rec) * 111
    min_dist = distances.min(axis=1)
    covered = min_dist <= distance_km
    return {
        'coverage_rate': covered.mean(),
        'covered': int(covered.sum()),
        'total': len(estate_df)
    }

@st.cache_data
def calculate_private_coverage(recycle_df, private_df, distance_km=0.5):
    if recycle_df.empty or private_df.empty:
        return {'coverage_rate': 0.0, 'covered': 0, 'total': 0, 'nearest_premium_ratio': 0.0}
    coords_rec = recycle_df[['latitude', 'longitude']].values
    coords_priv = private_df[['latitude', 'longitude']].values
    distances = cdist(coords_priv, coords_rec) * 111
    min_dist = distances.min(axis=1)
    covered_mask = min_dist <= distance_km
    nearest_idx = distances.argmin(axis=1)
    nearest_premium_ratio = 0.0
    if covered_mask.any():
        nearest_df = recycle_df.iloc[nearest_idx[covered_mask]]
        if 'is_premium' in nearest_df.columns:
            nearest_premium_ratio = nearest_df['is_premium'].mean()
    return {
        'coverage_rate': covered_mask.mean(),
        'covered': int(covered_mask.sum()),
        'total': len(private_df),
        'nearest_premium_ratio': nearest_premium_ratio
    }

# ====================== 識別未覆蓋建築物 / Identify Uncovered Buildings ======================
@st.cache_data
def get_uncovered_buildings(recycle_df, buildings_df, distance_km=0.3):
    if recycle_df.empty or buildings_df.empty:
        return pd.DataFrame()
    coords_rec = recycle_df[['latitude', 'longitude']].values
    coords_bld = buildings_df[['latitude', 'longitude']].values
    distances = cdist(coords_bld, coords_rec) * 111
    min_dist = distances.min(axis=1)
    uncovered_mask = min_dist > distance_km
    return buildings_df[uncovered_mask]

# ====================== 計算密度（添加空檢查和默認值） ======================
@st.cache_data
def calculate_density(df, radius_km=0.2):
    if df.empty:
        df['density'] = 0
        return df
    coords = df[['latitude', 'longitude']].values
    nbrs = NearestNeighbors(radius=radius_km / 111).fit(coords)
    distances, indices = nbrs.radius_neighbors(coords)
    df['density'] = [len(idx) - 1 for idx in indices]  # 排除自身
    return df

# ====================== 聚類並去重 ======================
@st.cache_data
def cluster_and_dedup(df, k):
    if df.empty or k == 0:
        return pd.DataFrame(columns=['latitude', 'longitude'])
    coords = df[['latitude', 'longitude']].values
    kmeans = KMeans(n_clusters=min(k, len(df)), random_state=42).fit(coords)
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=['latitude', 'longitude'])
    
    # 去重：合併 <0.2km 中心
    dist_matrix = cdist(centers.values, centers.values) * 111
    to_merge = np.triu(dist_matrix < 0.2, k=1)
    merge_groups = []
    visited = set()
    for i in range(len(centers)):
        if i in visited: continue
        group = [i]
        for j in range(i+1, len(centers)):
            if to_merge[i, j]:
                group.append(j)
                visited.add(j)
        merge_groups.append(group)
    
    dedup_centers = []
    for group in merge_groups:
        group_coords = centers.iloc[group]
        mean_center = group_coords.mean()
        dedup_centers.append(mean_center)
    
    return pd.DataFrame(dedup_centers, columns=['latitude', 'longitude'])

# ====================== 新增回收站建議（保持原邏輯） ======================
@st.cache_data
def propose_new_sites(uncovered_private, k=3, original_priv_cov=0.0):
    if uncovered_private.empty or k == 0:
        return pd.DataFrame(), 0.0, 0.0, 0.0
    coords = uncovered_private[['latitude', 'longitude']].values
    kmeans = KMeans(n_clusters=min(k, len(uncovered_private)), random_state=42).fit(coords)
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=['latitude', 'longitude'])
    centers['type'] = 'Recycle Station / Recycle Station'
    centers['is_premium'] = True
    new_coverage = calculate_private_coverage(centers, uncovered_private, 0.5)['coverage_rate']
    improvement = new_coverage * (1 - original_priv_cov) * 100
    new_total_cov = original_priv_cov * 100 + improvement
    efficiency = improvement / k if k > 0 else 0
    return centers, improvement, efficiency, new_total_cov

# ====================== 氣動系統規劃（改為全私宅 500 點） ======================
@st.cache_data
def plan_pneumatic_system(all_private_df, premium_df, original_priv_cov=0.0):
    if all_private_df.empty:
        return pd.DataFrame(), 0.0, 0.0, 0.0
    
    # 直接對所有私宅做 500 點聚類（包含已覆蓋區域）
    pneumatic_points = cluster_and_dedup(all_private_df, 500)
    pneumatic_points['type'] = 'Pneumatic Point / Pneumatic Point'
    pneumatic_points['is_premium'] = False
    
    # 連接最近高級站
    if not premium_df.empty:
        distances = cdist(pneumatic_points[['latitude', 'longitude']].values, 
                         premium_df[['latitude', 'longitude']].values)
        nearest_idx = distances.argmin(axis=1)
        pneumatic_points['nearest_premium_lat'] = premium_df.iloc[nearest_idx]['latitude'].values
        pneumatic_points['nearest_premium_lon'] = premium_df.iloc[nearest_idx]['longitude'].values
    
    # 評估：以 1km 為覆蓋半徑計算對私宅的覆蓋提升（較寬鬆評估）
    new_coverage = calculate_private_coverage(pneumatic_points, all_private_df, 1.0)['coverage_rate']
    improvement = (new_coverage - original_priv_cov) * 100
    new_total_cov = new_coverage * 100
    efficiency = improvement / len(pneumatic_points) if len(pneumatic_points) > 0 else 0
    
    return pneumatic_points, improvement, efficiency, new_total_cov

# ====================== 地圖生成函數（新增藍色星星高級站） ======================
def create_map(show_public, show_heat, filtered_recycle, filtered_public, filtered_private, 
               show_new_sites=False, new_sites=None, 
               show_pneumatic=False, pneumatic_points=None):
    m = folium.Map(
        location=[22.3, 114.1], 
        zoom_start=11, 
        tiles="CartoDB positron", 
        prefer_canvas=True,
        scrollWheelZoom=False,
        zoomControl=True,
        dragging=True
    )
    
    # 現有回收點（普通站用一般標記）
    if not filtered_recycle.empty:
        locations = filtered_recycle[['latitude', 'longitude']].values.tolist()
        popups = [
            f"{row.get('address_tc', '回收點 / Recycle Point') if not is_english else row.get('address_en', 'Recycle Point')}<br>"
            f"Type: {row['type']}<br>"
            f"Waste: {row.get('waste_types', 'Unknown')}"
            for _, row in filtered_recycle.iterrows()
        ]
        FastMarkerCluster(locations, popups=popups).add_to(m)
    
    # 高級站單獨用藍色星星
    premium_df = filtered_recycle[filtered_recycle['is_premium']]
    if not premium_df.empty:
        for _, row in premium_df.iterrows():
            folium.Marker(
                [row['latitude'], row['longitude']],
                popup=f"Premium Station<br>{row.get('address_tc', 'N/A') if not is_english else row.get('address_en', 'N/A')}",
                icon=folium.Icon(color='blue', icon='star')
            ).add_to(m)
    
    if show_public and not filtered_public.empty:
        for _, row in filtered_public.iterrows():
            folium.Marker(
                [row['latitude'], row['longitude']],
                popup=f"{row['estate_name']}<br>Units: {row.get('flats', 0):,}",
                icon=folium.Icon(color='red', icon='home')
            ).add_to(m)
    
    if show_heat and not filtered_private.empty:
        heat_data = filtered_private[['latitude', 'longitude']]
        HeatMap(heat_data.values.tolist(),
                radius=15, blur=20, gradient={0.2:'yellow', 0.5:'orange', 0.8:'red'}).add_to(m)
    
    # 新增回收站（綠色星星）
    if show_new_sites and new_sites is not None and not new_sites.empty:
        for _, site in new_sites.iterrows():
            folium.Marker(
                [site['latitude'], site['longitude']],
                popup="New Recycle Station",
                icon=folium.Icon(color='green', icon='star')
            ).add_to(m)
    
    # 氣動規劃層
    if show_pneumatic and pneumatic_points is not None and not pneumatic_points.empty:
        for _, point in pneumatic_points.iterrows():
            folium.CircleMarker(
                [point['latitude'], point['longitude']],
                radius=5, color='green', fill=False,
                popup="Pneumatic Point"
            ).add_to(m)
            if 'nearest_premium_lat' in point and not pd.isna(point['nearest_premium_lat']):
                folium.PolyLine(
                    locations=[[point['latitude'], point['longitude']], 
                               [point['nearest_premium_lat'], point['nearest_premium_lon']]],
                    color='purple', dash_array='5, 5'
                ).add_to(m)
    
    return m

# ====================== 主程序 / Main Function ======================
def main():
    st.markdown(f'<h1 class="main-header">{t["title"]}</h1>', unsafe_allow_html=True)
    
    with st.spinner(t["loading"]):
        df_recycle = load_recycling_points()
        df_public = load_public_housing()
        df_private = load_private_buildings()
    
    if df_recycle.empty:
        st.error(t["no_data"])
        return
    
    with st.expander(t["filters"], expanded=True):
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            selected_district = st.selectbox(t["district"], ['全部 / All'] + sorted(df_recycle['district'].unique().tolist()) if not is_english else ['All'] + sorted(df_recycle['district'].unique().tolist()))
        with col2:
            waste_options = ['全部 / All'] + sorted(set(t.strip() for types in df_recycle['waste_types'] if pd.notna(types) for t in str(types).split(','))) if not is_english else ['All'] + sorted(set(t.strip() for types in df_recycle['waste_types'] if pd.notna(types) for t in str(types).split(',')))
            selected_waste = st.selectbox(t["waste_type"], waste_options)
        with col3:
            selected_type = st.selectbox(t["station_type"], ['全部 / All'] + sorted(df_recycle['type'].unique().tolist()) if not is_english else ['All'] + sorted(df_recycle['type'].unique().tolist()))
        with col4:
            premium_only = st.checkbox(t["premium_only"], False)
        with col5:
            distance_m = st.slider(t["distance"], 100, 1000, 500, step=50)
            distance_km = distance_m / 1000.0
    
    filtered_recycle = df_recycle.copy()
    if selected_district != ('全部 / All' if not is_english else 'All'):
        filtered_recycle = filtered_recycle[filtered_recycle['district'] == selected_district]
    if selected_waste != ('全部 / All' if not is_english else 'All'):
        filtered_recycle = filtered_recycle[filtered_recycle['waste_types'].str.contains(selected_waste, na=False, case=False)]
    if selected_type != ('全部 / All' if not is_english else 'All'):
        filtered_recycle = filtered_recycle[filtered_recycle['type'] == selected_type]
    if premium_only:
        filtered_recycle = filtered_recycle[filtered_recycle['is_premium']]
    
    filtered_public = df_public[df_public['district'] == selected_district] if selected_district != ('全部 / All' if not is_english else 'All') else df_public
    filtered_private = df_private[df_private['district'] == selected_district] if selected_district != ('全部 / All' if not is_english else 'All') else df_private
    
    cov_pub = calculate_district_coverage(filtered_recycle, filtered_public, distance_km)
    cov_priv = calculate_private_coverage(filtered_recycle, filtered_private, distance_km)
    
    colA, colB, colC = st.columns(3)
    colA.metric(t["recycling_points"], f"{len(filtered_recycle):,}")
    colB.metric(t["public_coverage"], f"{cov_pub['coverage_rate']:.1%}")
    colC.metric(t["private_coverage"], f"{cov_priv['coverage_rate']:.1%}")
    
    st.markdown(f"## {t['private_title']}", unsafe_allow_html=True)
    st.metric(
        label=t["covered_buildings"],
        value=f"{cov_priv['covered']:,} / {cov_priv['total']:,}",
        delta=f"{cov_priv['coverage_rate']:.1%} {t['coverage_rate']}"
    )
    st.metric(t["premium_ratio"], f"{cov_priv['nearest_premium_ratio']:.1%}")
    
    # 距離覆蓋率趨勢圖
    if not filtered_recycle.empty and (not filtered_public.empty or not filtered_private.empty):
        st.subheader(t["distance_coverage_trend"])
        
        distance_points = list(range(100, 1050, 50))
        public_coverages = []
        private_coverages = []
        
        for d_m in distance_points:
            d_km = d_m / 1000.0
            pub_cov = calculate_district_coverage(filtered_recycle, filtered_public, d_km)['coverage_rate'] * 100
            priv_cov = calculate_private_coverage(filtered_recycle, filtered_private, d_km)['coverage_rate'] * 100
            public_coverages.append(pub_cov)
            private_coverages.append(priv_cov)
        
        trend_df = pd.DataFrame({
            "距離 (米) / Distance (m)" if not is_english else "Distance (m)": distance_points,
            "公屋覆蓋率 (%) / Public Estates (%)" if not is_english else "Public Estates Coverage (%)": public_coverages,
            "私樓覆蓋率 (%) / Private Buildings (%)" if not is_english else "Private Buildings Coverage (%)": private_coverages
        })
        
        st.line_chart(
            trend_df.set_index("距離 (米) / Distance (m)" if not is_english else "Distance (m)"),
            use_container_width=True,
            height=400
        )
        
        current_pub = cov_pub['coverage_rate'] * 100
        current_priv = cov_priv['coverage_rate'] * 100
        st.caption(f"目前 {distance_m} 米：公屋 {current_pub:.1f}% / 私樓 {current_priv:.1f}% / Current {distance_m} m: Public {current_pub:.1f}% / Private {current_priv:.1f}%" if not is_english else f"Current at {distance_m} m: Public {current_pub:.1f}% / Private {current_priv:.1f}%")
    else:
        st.info("無足夠資料繪製距離-覆蓋率趨勢圖 / Not enough data to plot distance-coverage trend." if not is_english else "Not enough data to plot distance-coverage trend.")

    st.subheader(t["common_waste"])
    waste_series = filtered_recycle['waste_types']
    if not waste_series.empty:
        prop = waste_series.value_counts(normalize=True).head(8) * 100
        st.bar_chart(prop, use_container_width=True)
        for wtype, pct in prop.items():
            st.write(f"• **{wtype}**：{pct:.1f}% / **{wtype}**: {pct:.1f}%" if not is_english else f"• **{wtype}**: {pct:.1f}%")
    else:
        st.write(t["no_data"])
    
    # ====================== 新增回收站建議 ======================
    st.markdown("---")
    with st.expander(t["new_sites_title"], expanded=False):
        st.markdown("""
        **原理解释 / Principle**：基於 K-means 聚類，針對私宅未覆蓋區生成建議位置，固定為高質量 Recycle Station。
        """ if not is_english else """
        **Principle**: Uses K-means clustering on uncovered private buildings to propose optimal new high-quality Recycle Station locations.
        """)
        uncovered_private = get_uncovered_buildings(filtered_recycle, filtered_private, 0.3)
        if not uncovered_private.empty:
            new_k = st.slider(t["new_sites_k"], min_value=0, max_value=50, value=3, step=1)
            new_sites, new_improvement, new_efficiency, new_total_cov = propose_new_sites(uncovered_private, new_k, cov_priv['coverage_rate'])
            st.metric(t["new_sites_impact"], f"原有 {cov_priv['coverage_rate']*100:.1f}% → 新 {new_total_cov:.1f}% (提升 {new_improvement:.1f}%)" if not is_english else f"From {cov_priv['coverage_rate']*100:.1f}% → {new_total_cov:.1f}% (Improvement: {new_improvement:.1f}%)")
            st.dataframe(new_sites)
        else:
            st.info("私宅無間隙區域 / No private gaps found." if not is_english else "No uncovered private building clusters found.")
    
    # ====================== 氣動系統規劃（全私宅 500 點） ======================
    with st.expander(t["pneumatic_title"], expanded=False):
        st.markdown("""
        **原理解释 / Principle**：對**全香港所有私宅**進行 K=500 聚類，生成氣動投放點（不再只限未覆蓋區），每個點連接最近高級站。  
        這樣可在覆蓋率已高的區域補強密度，也能在低覆蓋區填補空白，達到更全面的私宅服務網絡。
        
        **主要公式 / Key Formula**：
        - 聚類中心 = K-means (K=500) 最小化 ∑(距離²)
        - 去重 = 合併 <200m 中心
        - 覆蓋評估半徑 = 1km（較寬鬆）
        - 提升 = (新覆蓋率 - 原有私宅覆蓋率) × 100%
        """ if not is_english else """
        **Principle**: Performs K=500 clustering on **all private buildings in Hong Kong** to generate 500 pneumatic collection points (not limited to uncovered areas). Each point connects to the nearest premium station.

        **Key Formula**:
        - Cluster centers = K-means (K=500) minimizing sum of squared distances
        - Deduplication = merge centers < 200m apart
        - Coverage evaluation radius = 1 km (lenient)
        - Improvement = (New Coverage Rate - Original Private Coverage) × 100%
        """)
        st.caption(t["pneumatic_k"])
        
        # 使用全部私宅（不限未覆蓋）
        all_private = filtered_private.copy()
        premium_stations = filtered_recycle[filtered_recycle['is_premium']]
        
        if not all_private.empty and not premium_stations.empty:
            pneumatic_points, pneumatic_improvement, pneumatic_efficiency, pneumatic_total_cov = plan_pneumatic_system(
                all_private, premium_stations, cov_priv['coverage_rate']
            )
            st.metric(t["pneumatic_impact"], 
                      f"原有 {cov_priv['coverage_rate']*100:.1f}% → 新 {pneumatic_total_cov:.1f}% (提升約 {pneumatic_improvement:+.1f}%)" if not is_english else 
                      f"From {cov_priv['coverage_rate']*100:.1f}% → {pneumatic_total_cov:.1f}% (Improvement: {pneumatic_improvement:+.1f}%)")
            st.caption(f"效率約：{pneumatic_efficiency:.3f}% 覆蓋 / 每點" if not is_english else f"Approx. efficiency: {pneumatic_efficiency:.3f}% coverage gain per point")
            st.dataframe(pneumatic_points[['latitude', 'longitude']].head(10))
            st.caption("（完整 500 點已繪製於地圖，表格僅展示前10筆）" if not is_english else "(All 500 points are plotted on map; table shows first 10 only)")
        else:
            st.info("無私宅資料或無高級站 / No private buildings or premium stations." if not is_english else "No private buildings data or no premium stations available.")
    
    st.subheader(t["map_title"])
    
    show_public = st.checkbox(t["show_public"], value=False)
    show_heat = st.checkbox(t["show_private_heat"], value=True)
    
    col_map1, col_map2 = st.columns([1,3])
    with col_map1:
        show_new_sites_layer = st.checkbox(t["show_new_sites"])
        st.caption(t["legend_new_sites"])
    
    with col_map2:
        show_pneumatic_layer = st.checkbox(t["show_pneumatic"], value=False)
        st.caption(t["legend_pneumatic"])
        st.caption(t["legend_premium"])
    
    m = create_map(
        show_public, show_heat, filtered_recycle, filtered_public, filtered_private,
        show_new_sites=show_new_sites_layer, new_sites=locals().get('new_sites'),
        show_pneumatic=show_pneumatic_layer, pneumatic_points=locals().get('pneumatic_points')
    )
    folium_static(m, width=1200, height=650)
    
    st.markdown("---")

    # ====================== 新增：研究方法、公式與原理（中英雙語） ======================
    with st.expander("研究方法、评价函数与公式 / Research Methods, Evaluation Functions & Formulas" if not is_english else "Research Methods, Evaluation Functions & Formulas", expanded=False):
        st.markdown("""
        ### 核心研究方法 / Core Research Methods

        1. **空間覆蓋分析 (Spatial Coverage Analysis)**  
           使用歐幾里得距離（球面近似）判斷建築是否在回收站覆蓋範圍內。  
           *Principle*: 計算每個建築到最近回收站的距離，若 ≤ 覆蓋半徑則視為已覆蓋。

        2. **K-means 聚類 (K-means Clustering)**  
           用於建議新增回收站位置及氣動系統投放點規劃。  
           *Principle*: 最小化各點到其所屬聚類中心的平方距離總和，實現空間最優分組。

        3. **最近鄰匹配與去重 (Nearest Neighbor Matching & Deduplication)**  
           氣動點連接到最近高級站；聚類中心 < 200m 合併。  
           *Principle*: 避免過度密集佈點，提升系統實際可行性。

        4. **熱力圖與密度估計 (Heatmap & Density Estimation)**  
           展示私樓分佈密度（Kernel Density Estimation 近似）。  
           *Principle*: 視覺化高密度區域，輔助識別服務缺口。

        ### 主要評價函數與公式 / Key Evaluation Functions & Formulas

        **覆蓋率 (Coverage Rate)**  
        $$
        \\text{Coverage Rate} = \\frac{\\text{Number of buildings with } d_{\\min} \\leq r}{\\text{Total number of buildings}} \\times 100\\%
        $$  
        其中 \( d_{\\min} \) 為建築到最近回收站的距離（km），\( r \) 為覆蓋半徑（預設 0.5 km）。

        **覆蓋提升 (Coverage Improvement)**  
        $$
        \\Delta \\text{Coverage} = (\\text{New Coverage Rate} - \\text{Original Coverage Rate}) \\times 100\\%
        $$

        **每點效率 (Efficiency per Point)**  
        $$
        \\text{Efficiency} = \\frac{\\Delta \\text{Coverage}}{\\text{Number of new points}}
        $$

        **氣動系統覆蓋評估**  
        使用較寬鬆半徑 1 km 計算覆蓋率（模擬氣動系統更大服務範圍）。

        **最近優質站比例 (Nearest Premium Ratio)**  
        在已覆蓋建築中，最近站點為優質站（GREEN@COMMUNITY / Recycling Station）的比例。

        ### 原理總結 / Summary of Principles

        - **第一哩問題 (First-mile Problem)**：私宅居民投放不便 → 透過新增站點 + 氣動系統縮短距離，提升參與意願。
        - **全域優化而非局部填補**：氣動點對全私宅進行 K=500 聚類，兼顧高密度區補強與低覆蓋區填補。
        - **數據驅動決策**：所有建議位置均基於真實建築坐標與現有設施，確保科學性與可操作性。
        """ if not is_english else """
        ### Core Research Methods

        1. **Spatial Coverage Analysis**  
           Uses Euclidean distance (spherical approximation) to determine if a building is within coverage range of a recycling point.

        2. **K-means Clustering**  
           Applied to propose new station locations and pneumatic point planning.

        3. **Nearest Neighbor Matching & Deduplication**  
           Pneumatic points connect to nearest premium station; cluster centers < 200m are merged.

        4. **Heatmap & Density Estimation**  
           Visualizes private building density distribution.

        ### Key Evaluation Functions & Formulas

        **Coverage Rate**  
        $$
        \\text{Coverage Rate} = \\frac{\\text{Number of buildings with } d_{\\min} \\leq r}{\\text{Total number of buildings}} \\times 100\\%
        $$

        **Coverage Improvement**  
        $$
        \\Delta \\text{Coverage} = (\\text{New Coverage Rate} - \\text{Original Coverage Rate}) \\times 100\\%
        $$

        **Efficiency per Point**  
        $$
        \\text{Efficiency} = \\frac{\\Delta \\text{Coverage}}{\\text{Number of new points}}
        $$

        **Pneumatic Coverage Evaluation**  
        Uses a lenient 1 km radius to simulate larger service range of pneumatic system.

        **Nearest Premium Ratio**  
        Proportion of covered buildings whose nearest station is a premium one.

        ### Summary of Principles

        - Addresses first-mile problem for private estates.
        - Global optimization instead of local gap-filling.
        - All proposals are data-driven using real building coordinates and existing facilities.
        """)

    # ====================== 数据来源（基于官方及 CUHK Data Hack 2026） ======================
    st.markdown("### 数据来源 / Data Sources" if not is_english else "### Data Sources")
    st.markdown("""
    - **Recyclable Collection Points Data**  
      香港各區回收收集點基礎設施數據  
      Source: Environmental Protection Department (EPD), Hong Kong  
      Link: https://data.gov.hk/en-data/dataset/hk-epd-recycteam-waste-less-recyclable-collection-points-data

    - **Location and Profile of Public Housing Estates**  
      公屋位置、概況及單位數量  
      Source: Hong Kong Housing Authority (via data.gov.hk)  
      Link: https://www.housingauthority.gov.hk/datagovhk/prh-estates.json

    - **Database of Private Buildings in Hong Kong**  
      香港私樓綜合數據庫（含坐標，用於私宅覆蓋分析）  
      Source: Home Affairs Department / data.gov.hk  
      Link: https://data.gov.hk/en-data/dataset/hk-had-json1-db-of-private-buildings-in-hong-kong

    - **Open Space Database of Recycling Station**  
      GREEN@COMMUNITY 回收站空間數據  
      Source: Environmental Protection Department (EPD)  
      Link: https://data.gov.hk/en-data/dataset/hk-epd-wrrteam-recycling-station

    - **Additional References**  
      - 2021 Population Census (人口普查 - 用於密度與分區分析)  
        Source: Census and Statistics Department  
      - Waste Management Facilities & Historical Recovery Trends  
        Source: EPD Annual Reports & data.gov.hk

    **主要數據門戶 / Primary Data Portal**  
    Hong Kong Government Open Data Portal: https://data.gov.hk  
    CUHK Data Hack 2026 官方頁面: https://libguides.lib.cuhk.edu.hk/datahack/2026-data

    **數據存取日期 / Data Access Date**: March 2026  
    **免責聲明 / Disclaimer**: 所有數據來自香港政府公開數據平台，僅用於 CUHK Data Hack 2026 比賽（Green Loop Challenge），非商業用途。
    """ if not is_english else """
    - **Recyclable Collection Points Data**  
      Source: Environmental Protection Department (EPD), Hong Kong  
      Link: https://data.gov.hk/en-data/dataset/hk-epd-recycteam-waste-less-recyclable-collection-points-data

    - **Location and Profile of Public Housing Estates**  
      Source: Hong Kong Housing Authority (via data.gov.hk)  
      Link: https://www.housingauthority.gov.hk/datagovhk/prh-estates.json

    - **Database of Private Buildings in Hong Kong**  
      Source: Home Affairs Department / data.gov.hk  
      Link: https://data.gov.hk/en-data/dataset/hk-had-json1-db-of-private-buildings-in-hong-kong

    - **Open Space Database of Recycling Station**  
      Source: Environmental Protection Department (EPD)  
      Link: https://data.gov.hk/en-data/dataset/hk-epd-wrrteam-recycling-station

    - **Additional References**  
      - 2021 Population Census  
      - EPD Annual Reports & historical recovery trends

    **Primary Data Portal**  
    Hong Kong Government Open Data Portal: https://data.gov.hk  
    CUHK Data Hack 2026: https://libguides.lib.cuhk.edu.hk/datahack/2026-data

    **Data Access Date**: March 2026  
    **Disclaimer**: All data from Hong Kong government open data portal, used solely for CUHK Data Hack 2026 (Green Loop Challenge), non-commercial purpose.
    """)

if __name__ == '__main__':
    main()
