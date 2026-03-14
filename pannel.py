# panel.py - 最终完整版（性能优化版）
# 优化内容：
# 1. 添加内存监控和垃圾回收
# 2. 优化大数据加载方式（分块加载）
# 3. 减少不必要的计算
# 4. 添加数据缓存策略
# 5. 优化地理计算性能

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
import plotly.express as px
import gc
import psutil
import warnings
warnings.filterwarnings('ignore')

# ====================== 頁面配置 ======================
st.set_page_config(
    page_title="香港回收網絡優化 - Green Loop Challenge",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== 内存监控 ======================
def get_memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # 转换为 MB

def log_memory_usage(step_name):
    """记录内存使用"""
    memory_used = get_memory_usage()
    st.sidebar.write(f"📊 {step_name}: {memory_used:.1f} MB")
    return memory_used

# ====================== 數據清理函數 ======================
def cleanup_memory():
    """清理内存"""
    gc.collect()
    for i in range(3):
        gc.collect(i)

# ====================== 語言切換 ======================
with st.sidebar:
    lang = st.selectbox("語言 / Language", ["中文", "English"], index=0)
    if st.checkbox("顯示調試信息 / Show Debug Info", False):
        show_debug = True
        st.write(f"初始内存: {get_memory_usage():.1f} MB")
    else:
        show_debug = False

# ====================== 文字字典 ======================
texts = {
    "中文": {
        "title": "♻️ 香港回收網絡優化 - Green Loop Challenge",
        "loading": "正在載入資料...",
        "filters": "🔍 篩選條件",
        "district": "地區",
        "waste_type": "回收類型",
        "station_type": "回收點類型",
        "premium_only": "僅優質站點",
        "distance": "覆蓋距離 (米)",
        "show_public": "顯示公屋標記（紅色）",
        "show_private_heat": "顯示私樓密度熱力圖",
        "private_title": "選定地區覆蓋分析",
        "covered_buildings": "符合條件的建築數",
        "coverage_rate": "覆蓋率",
        "of_total": "佔總數",
        "premium_ratio": "最近為優質站比例",
        "common_waste": "垃圾回收種類比例",
        "formula_title": "覆蓋率計算公式",
        "recycling_points": "回收點",
        "public_coverage": "公屋覆蓋率",
        "private_coverage": "私樓覆蓋率",
        "map_title": "全香港回收網絡地圖",
        "proposal": "數據驅動提案",
        "insight": "核心洞察",
        "suggestion": "建議",
        "impact": "預期影響",
        "no_data": "無符合條件資料",
        "distance_coverage_trend": "覆蓋率隨距離變化趨勢",
        "new_sites_title": "新增回收站点建议",
        "new_sites_k": "建议站点数量",
        "new_sites_impact": "预计覆盖提升",
        "pneumatic_title": "气动系统规划",
        "pneumatic_formula": "规划公式",
        "pneumatic_mini_k": "小型中心数量",
        "pneumatic_impact": "预计覆盖提升"
    },
    "English": {
        "title": "♻️ Hong Kong Recycling Network Optimization",
        "loading": "Loading data...",
        "filters": "🔍 Filter Conditions",
        "district": "District",
        "waste_type": "Waste Type",
        "station_type": "Station Type",
        "premium_only": "Premium Stations Only",
        "distance": "Coverage Distance (meters)",
        "show_public": "Show Public Estates Markers (Red)",
        "show_private_heat": "Show Private Buildings Density Heatmap",
        "private_title": "Selected District Coverage Analysis",
        "covered_buildings": "Buildings Covered",
        "coverage_rate": "Coverage Rate",
        "of_total": "of Total",
        "premium_ratio": "Nearest are Premium Stations",
        "common_waste": "Waste Type Proportions",
        "formula_title": "Coverage Rate Formula",
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
        "new_sites_k": "Number of Proposed Sites",
        "new_sites_impact": "Expected Coverage Improvement",
        "pneumatic_title": "Pneumatic System Planning",
        "pneumatic_formula": "Planning Formula",
        "pneumatic_mini_k": "Number of Mini Centers",
        "pneumatic_impact": "Expected Coverage Improvement"
    }
}
t = texts[lang]

# ====================== 地區名稱標準化 ======================
@st.cache_data
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

# ====================== 優化後的數據加載 ======================
@st.cache_data(ttl=3600, max_entries=2)
def load_recycling_points():
    """加载回收点数据（优化版）"""
    try:
        # 只讀取需要的列
        usecols = ['cp_id', 'cp_state', 'district_id', 'address_en', 'address_tc',
                   'lat', 'lgt', 'waste_type', 'legend', 'openhour_tc']
        
        df = pd.read_csv("Recyclable-Collection-Points-Data.csv", 
                         encoding='utf-8',
                         usecols=usecols,
                         low_memory=False)
        
        df.columns = df.columns.str.strip()
        
        # 重命名列
        mapping = {
            'cp_id': 'id', 'cp_state': 'status', 'district_id': 'district',
            'address_en': 'address_en', 'address_tc': 'address_tc',
            'lat': 'latitude', 'lgt': 'longitude', 'waste_type': 'waste_types',
            'legend': 'type', 'openhour_tc': 'hours_tc'
        }
        df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
        
        # 快速过滤
        df = df.dropna(subset=['latitude', 'longitude'])
        df = df[(df['latitude'] != 0) & (df['longitude'] != 0)]
        df = df[(df['latitude'].between(22.15, 22.55)) & (df['longitude'].between(113.8, 114.4))]
        
        # 填充缺失值
        df['waste_types'] = df['waste_types'].fillna('Unknown')
        df['type'] = df.get('type', '回收站').fillna('回收站')
        df['district'] = df.get('district', 'Unknown').fillna('Unknown')
        
        # 創建優質站點標記
        df['is_premium'] = df['type'].str.contains('GREEN@COMMUNITY|Recycling Station|Recycling Store', 
                                                   case=False, na=False)
        
        # 標準化地區名稱
        df['district'] = df['district'].apply(normalize_district_name)
        
        # 選擇需要的列
        keep_cols = ['id', 'district', 'latitude', 'longitude', 'waste_types', 
                     'type', 'is_premium', 'address_en', 'address_tc']
        df = df[[col for col in keep_cols if col in df.columns]]
        
        return df
        
    except Exception as e:
        st.error(f"回收點 CSV 加載失敗: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, max_entries=2)
def load_public_housing():
    """加载公屋数据（优化版）"""
    path = "prh-estates.json"
    if not os.path.exists(path): 
        return pd.DataFrame()
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        estates = []
        # 限制讀取數量以節省內存
        max_items = 2000
        
        for i, item in enumerate(data if isinstance(data, list) else [data]):
            if i >= max_items:
                break
                
            estate_name = item.get("Estate Name", {}).get("en", "Unknown")
            district = item.get("District Name", {}).get("en", "Unknown")
            lat = item.get("Estate Map Latitude")
            lon = item.get("Estate Map Longitude")
            
            if lat is None or lon is None: 
                continue
            
            # 優化 flats 提取
            flats = 0
            flats_str = str(item.get("No. of Units", item.get("No. of Rental Flats", "0")))
            match = re.search(r'\d[\d\s,]*', flats_str)
            if match:
                flats = int(match.group().replace(" ", "").replace(",", ""))
            
            estates.append({
                'estate_name': estate_name, 
                'district': district, 
                'flats': flats,
                'latitude': lat, 
                'longitude': lon,
                'weight': flats if flats > 0 else 1  # 使用 flats 作為權重
            })
        
        df = pd.DataFrame(estates)
        df['district'] = df['district'].apply(normalize_district_name)
        
        return df
        
    except Exception as e:
        st.warning(f"公屋載入失敗: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, max_entries=2)
def load_private_buildings():
    """加载私楼数据（优化版 - 分块读取）"""
    path = "PrivateBuildings.csv"
    if not os.path.exists(path):
        st.error(f"私宅文件不存在: {path}")
        return pd.DataFrame()
    
    try:
        # 分块读取大型 CSV
        chunks = []
        chunk_size = 10000  # 每块1万行
        
        for chunk in pd.read_csv(path, 
                                usecols=['LATITUDE', 'LONGITUDE', 'SEARCH1_E'],
                                encoding='utf-8-sig',
                                chunksize=chunk_size,
                                on_bad_lines='skip'):
            
            chunk.columns = chunk.columns.str.strip()
            chunk = chunk.rename(columns={'SEARCH1_E': 'district', 
                                          'LATITUDE': 'latitude', 
                                          'LONGITUDE': 'longitude'})
            
            # 立即过滤无效数据
            chunk = chunk.dropna(subset=['latitude', 'longitude'])
            chunk = chunk[(chunk['latitude'].between(22.0, 22.6)) & 
                         (chunk['longitude'].between(113.8, 114.5))]
            
            if not chunk.empty:
                chunk['type'] = 'Private Building'
                chunk['district'] = chunk['district'].apply(normalize_district_name)
                chunk['weight'] = 50  # 固定权重
                chunks.append(chunk)
            
            # 清理内存
            del chunk
            cleanup_memory()
        
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
            # 釋放 chunks 內存
            del chunks
            cleanup_memory()
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"私宅載入失敗: {str(e)}")
        return pd.DataFrame()

# ====================== 優化後的覆蓋率計算 ======================
@st.cache_data(ttl=300)
def calculate_district_coverage(recycle_df, estate_df, distance_km=0.5):
    """計算覆蓋率（優化版）"""
    if recycle_df.empty or estate_df.empty:
        return {'coverage_rate': 0.0, 'covered': 0, 'total': 0}
    
    # 抽樣計算（如果數據太大）
    if len(estate_df) > 10000:
        estate_sample = estate_df.sample(n=10000, random_state=42)
    else:
        estate_sample = estate_df
    
    coords_rec = recycle_df[['latitude', 'longitude']].values
    coords_est = estate_sample[['latitude', 'longitude']].values
    
    # 優化距離計算
    distances = cdist(coords_est, coords_rec) * 111
    min_dist = distances.min(axis=1)
    covered = (min_dist <= distance_km).sum()
    
    return {
        'coverage_rate': covered / len(estate_sample),
        'covered': covered,
        'total': len(estate_df)  # 返回總數而不是抽樣數
    }

def calculate_private_coverage(recycle_df, private_df, distance_km=0.5):
    """計算私樓覆蓋率（優化版）"""
    if recycle_df.empty or private_df.empty:
        return {'coverage_rate': 0.0, 'covered': 0, 'total': 0,
                'nearest_premium_ratio': 0.0}
    
    # 抽樣計算
    if len(private_df) > 10000:
        private_sample = private_df.sample(n=10000, random_state=42)
    else:
        private_sample = private_df
    
    coords_rec = recycle_df[['latitude', 'longitude']].values
    coords_priv = private_sample[['latitude', 'longitude']].values
    
    # 使用 numpy 向量化計算
    distances = cdist(coords_priv, coords_rec) * 111
    min_dist = distances.min(axis=1)
    covered_mask = min_dist <= distance_km
    
    # 計算優質站點比例
    nearest_idx = distances.argmin(axis=1)
    nearest_premium_ratio = recycle_df.iloc[nearest_idx[covered_mask]]['is_premium'].mean() if covered_mask.any() else 0.0
    
    return {
        'coverage_rate': covered_mask.mean(),
        'covered': int(covered_mask.sum()),
        'total': len(private_df),
        'nearest_premium_ratio': nearest_premium_ratio
    }

# ====================== 識別未覆蓋建築物（優化版） ======================
@st.cache_data(ttl=300)
def get_uncovered_buildings(recycle_df, buildings_df, distance_km=0.3):
    """識別未覆蓋建築（優化版）"""
    if recycle_df.empty or buildings_df.empty:
        return pd.DataFrame()
    
    # 限制計算數量
    if len(buildings_df) > 20000:
        buildings_sample = buildings_df.sample(n=20000, random_state=42)
    else:
        buildings_sample = buildings_df
    
    coords_rec = recycle_df[['latitude', 'longitude']].values
    coords_bld = buildings_sample[['latitude', 'longitude']].values
    
    # 使用 KDTree 優化距離計算（如果數據量大）
    try:
        from scipy.spatial import KDTree
        tree = KDTree(coords_rec * 111)  # 轉換為公里
        distances, _ = tree.query(coords_bld * 111)
        uncovered_mask = distances > distance_km
    except:
        # 備用方法
        distances = cdist(coords_bld, coords_rec) * 111
        min_dist = distances.min(axis=1)
        uncovered_mask = min_dist > distance_km
    
    return buildings_sample[uncovered_mask]

# ====================== 地圖生成函數 ======================
@st.cache_data(ttl=60)
def create_map_data(filtered_recycle, filtered_public, filtered_private):
    """準備地圖數據（緩存）"""
    return {
        'recycle': filtered_recycle.to_dict('records'),
        'public': filtered_public.to_dict('records'),
        'private': filtered_private.to_dict('records')
    }

def create_map(show_public, show_heat, filtered_recycle, filtered_public, filtered_private, 
               new_sites=None, mini_centers=None, central=None):
    """生成地圖（優化版）"""
    m = folium.Map(
        location=[22.3, 114.1], 
        zoom_start=11, 
        tiles="CartoDB positron", 
        prefer_canvas=True,
        scrollWheelZoom=False,
        zoomControl=True,
        dragging=True
    )
    
    # 限制地圖元素數量
    if not filtered_recycle.empty:
        max_markers = 500
        recycle_data = filtered_recycle.head(max_markers)
        
        # 使用批量添加標記
        marker_cluster = FastMarkerCluster([]).add_to(m)
        for _, row in recycle_data.iterrows():
            popup = f"{row.get('address_tc', '回收點')}<br>類型: {row['type']}"
            folium.Marker(
                [row['latitude'], row['longitude']],
                popup=popup,
                icon=folium.Icon(color='green', icon='info-sign', prefix='fa')
            ).add_to(marker_cluster)
    
    if show_public and not filtered_public.empty:
        public_data = filtered_public.head(200)  # 限制公屋顯示數量
        for _, row in public_data.iterrows():
            folium.CircleMarker(
                [row['latitude'], row['longitude']],
                radius=3,
                color='red',
                fill=True,
                popup=row.get('estate_name', '公屋'),
                fillOpacity=0.6
            ).add_to(m)
    
    if show_heat and not filtered_private.empty:
        # 限制熱力圖數據點
        heat_data = filtered_private.sample(n=min(5000, len(filtered_private)))[['latitude', 'longitude']]
        HeatMap(heat_data.values.tolist(),
                radius=10, blur=15, 
                gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'red'}).add_to(m)
    
    # 新站點標記（數量少，不限制）
    if new_sites is not None and not new_sites.empty:
        for _, site in new_sites.iterrows():
            folium.Marker(
                [site['latitude'], site['longitude']],
                popup=f"新站點: {site['type']}",
                icon=folium.Icon(color='green', icon='star')
            ).add_to(m)
    
    return m

# ====================== 主程序 ======================
def main():
    st.markdown(f'# {t["title"]}')
    
    if show_debug:
        log_memory_usage("開始")
    
    # 數據加載
    with st.spinner(t["loading"]):
        df_recycle = load_recycling_points()
        df_public = load_public_housing()
        df_private = load_private_buildings()
    
    if show_debug:
        log_memory_usage("數據加載完成")
        st.sidebar.write(f"回收點: {len(df_recycle)}")
        st.sidebar.write(f"公屋: {len(df_public)}")
        st.sidebar.write(f"私樓: {len(df_private)}")
    
    if df_recycle.empty:
        st.error(t["no_data"])
        return
    
    # 篩選條件
    with st.sidebar:
        st.header(t["filters"])
        
        districts = ['全部 / All'] + sorted(df_recycle['district'].unique().tolist())
        selected_district = st.selectbox(t["district"], districts)
        
        waste_types = ['全部 / All'] + sorted(set(
            t.strip() for types in df_recycle['waste_types'] 
            if pd.notna(types) for t in str(types).split(',')
        ))
        selected_waste = st.selectbox(t["waste_type"], waste_types)
        
        station_types = ['全部 / All'] + sorted(df_recycle['type'].unique().tolist())
        selected_type = st.selectbox(t["station_type"], station_types)
        
        premium_only = st.checkbox(t["premium_only"], False)
        
        distance_m = st.slider(t["distance"], 100, 1000, 500, step=50)
        distance_km = distance_m / 1000.0
    
    # 篩選數據
    filtered_recycle = df_recycle.copy()
    if selected_district != '全部 / All':
        filtered_recycle = filtered_recycle[filtered_recycle['district'] == selected_district]
    if selected_waste != '全部 / All':
        filtered_recycle = filtered_recycle[filtered_recycle['waste_types'].str.contains(selected_waste, na=False, case=False)]
    if selected_type != '全部 / All':
        filtered_recycle = filtered_recycle[filtered_recycle['type'] == selected_type]
    if premium_only:
        filtered_recycle = filtered_recycle[filtered_recycle['is_premium']]
    
    # 地區過濾
    mask = df_public['district'] == selected_district if selected_district != '全部 / All' else slice(None)
    filtered_public = df_public[mask] if not df_public.empty else df_public
    
    mask = df_private['district'] == selected_district if selected_district != '全部 / All' else slice(None)
    filtered_private = df_private[mask] if not df_private.empty else df_private
    
    if show_debug:
        log_memory_usage("篩選後")
    
    # 計算覆蓋率
    cov_pub = calculate_district_coverage(filtered_recycle, filtered_public, distance_km)
    cov_priv = calculate_private_coverage(filtered_recycle, filtered_private, distance_km)
    
    # 顯示指標
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(t["recycling_points"], f"{len(filtered_recycle):,}")
    with col2:
        st.metric(t["public_coverage"], f"{cov_pub['coverage_rate']:.1%}")
    with col3:
        st.metric(t["private_coverage"], f"{cov_priv['coverage_rate']:.1%}")
    
    # 詳細分析
    with st.expander(t["private_title"], expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(
                label=t["covered_buildings"],
                value=f"{cov_priv['covered']:,} / {cov_priv['total']:,}",
                delta=f"{cov_priv['coverage_rate']:.1%}"
            )
        with col_b:
            st.metric(t["premium_ratio"], f"{cov_priv['nearest_premium_ratio']:.1%}")
    
    # 垃圾類型分佈
    if not filtered_recycle.empty:
        with st.expander(t["common_waste"], expanded=False):
            waste_counts = filtered_recycle['waste_types'].value_counts().head(10)
            st.bar_chart(waste_counts)
    
    # 新站點建議
    with st.sidebar:
        st.markdown("---")
        if st.checkbox("🔧 新站點規劃 / Site Planning", False):
            st.info("正在計算未覆蓋區域...")
            
            # 合併建築物
            all_buildings = pd.concat([
                filtered_public.assign(weight=filtered_public.get('flats', 50)),
                filtered_private.assign(weight=50)
            ]) if not filtered_public.empty or not filtered_private.empty else pd.DataFrame()
            
            if not all_buildings.empty:
                uncovered = get_uncovered_buildings(filtered_recycle, all_buildings, 0.3)
                
                if not uncovered.empty:
                    k = st.slider("K (站點數量)", 1, 10, 3)
                    
                    if st.button("計算新站點"):
                        with st.spinner("計算中..."):
                            # 簡單聚類
                            from sklearn.cluster import KMeans
                            coords = uncovered[['latitude', 'longitude']].values
                            
                            # 限制數據量
                            if len(coords) > 5000:
                                coords = coords[np.random.choice(len(coords), 5000, replace=False)]
                            
                            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                            kmeans.fit(coords)
                            
                            new_sites = pd.DataFrame(kmeans.cluster_centers_, 
                                                    columns=['latitude', 'longitude'])
                            new_sites['type'] = 'Proposed Site'
                            
                            st.session_state['new_sites'] = new_sites
                            st.success(f"已找到 {k} 個建議站點")
                            
                            # 計算覆蓋提升
                            improved = calculate_private_coverage(
                                pd.concat([filtered_recycle, new_sites[['latitude', 'longitude']].assign(is_premium=True)]),
                                uncovered, 0.5
                            )['coverage_rate']
                            
                            st.metric("預期覆蓋提升", f"{improved:.1%}")
    
    # 地圖顯示
    st.subheader(t["map_title"])
    
    col1, col2 = st.columns(2)
    with col1:
        show_public = st.checkbox(t["show_public"], value=False)
    with col2:
        show_heat = st.checkbox(t["show_private_heat"], value=True)
    
    # 獲取新站點（如果存在）
    new_sites = st.session_state.get('new_sites', None)
    
    # 生成地圖
    m = create_map(
        show_public, 
        show_heat, 
        filtered_recycle.head(1000),  # 限制地圖顯示數量
        filtered_public.head(200) if show_public else pd.DataFrame(),
        filtered_private.head(5000) if show_heat else pd.DataFrame(),
        new_sites=new_sites
    )
    
    folium_static(m, width=1200, height=600)
    
    if show_debug:
        log_memory_usage("完成")
        cleanup_memory()

if __name__ == '__main__':
    main()
