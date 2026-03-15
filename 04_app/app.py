# -*- coding: utf-8 -*-
"""
음악 구독 서비스 이탈 예측 대시보드 (Clear Pastel White & Pink Edition)
실행: streamlit run 04_app/app.py
"""

import sys
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------
# 🚨 src 폴더 인식 및 전처리 모듈 로드
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.clustering import make_rfm_table

# ---------------------------------------------------------
# 1. 페이지 설정 및 맑은 핑크 CSS
# ---------------------------------------------------------
st.set_page_config(page_title="음악 구독 이탈 예측 대시보드", page_icon="🎵", layout="wide")

st.markdown(
    """
<style>
/* 🌸 탁하지 않은 '맑고 투명한 파스텔 핑크' 그라데이션 조명 🌸 */
.stApp { background:
    radial-gradient(1100px 520px at 18% 10%, rgba(255, 94, 148, 0.10), transparent 55%),
    radial-gradient(900px 420px at 82% 0%, rgba(255, 94, 148, 0.06), transparent 55%),
    #FFFFFF; 
color: #1F2937; }
.block-container { padding-top: 2.5rem !important; padding-bottom: 2.0rem !important; max-width: 1200px; }
section[data-testid="stSidebar"] > div { background-color: #FDFDFD; border-right: 1px solid #F3F4F6; }

.hero-title { font-size: 2.3rem; font-weight: 850; letter-spacing: -0.02em; margin: 0 0 0.35rem 0; color: #111827; }
.hero-subtitle { font-size: 1.05rem; color: #4B5563; margin: 0 0 2rem 0; }

/* 밝고 부드러운 카드 디자인 */
.landing-card, .card { background-color: #FFFFFF; border: 1px solid #F3F4F6; border-radius: 16px; padding: 20px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02); transition: transform 0.2s ease, box-shadow 0.2s ease; }
.landing-card:hover { transform: translateY(-3px); box-shadow: 0 8px 15px rgba(255, 94, 148, 0.15); border-color: #FF5E94; }
.landing-card .title { font-size: 1.2rem; font-weight: 800; margin: 0 0 0.6rem 0; color: #111827; }
.landing-card .desc { font-size: 0.95rem; color: #6B7280; line-height: 1.5; }

.card { display:flex; flex-direction:column; gap:6px; min-height: 100px; justify-content: center;}
.card .k { font-size: 0.85rem !important; font-weight: 700 !important; color: #6B7280 !important; }
.card .v { font-size: 1.7rem !important; font-weight: 800 !important; color: #111827 !important; line-height: 1.1; }

/* 🌸 채도가 높고 맑은 캔디 핑크 버튼 🌸 */
div.stButton > button { border-radius: 12px; padding: 0.6rem 1.5rem; font-weight: 700; border: 1px solid #E5E7EB; background-color: #FFFFFF; color: #374151; transition: all 0.2s ease; }
div.stButton > button:hover { background-color: #FFF0F5; border-color: #FF5E94; color: #E11D48; }
div.stButton > button[kind="primary"] { border: none !important; background: linear-gradient(135deg, #FF85AD, #FF5E94) !important; color: #FFFFFF !important; box-shadow: 0 4px 10px rgba(255, 94, 148, 0.3); width: 100%;}
div.stButton > button[kind="primary"]:hover { transform: scale(1.02); filter: brightness(1.05); color: #FFFFFF !important;}

/* 🌸 슬라이더 색상 (맑은 핑크) 🌸 */
.stSlider div[data-baseweb="slider"] div[role="slider"] { background-color: #FF5E94 !important; border: 2px solid #FFFFFF !important; box-shadow: 0 2px 6px rgba(255, 94, 148, 0.4) !important; }
.stSlider div[data-baseweb="slider"] div[role="slider"]:focus { box-shadow: 0 0 0 0.2rem rgba(255, 94, 148, 0.25) !important; outline: none !important; }
.stSlider div[data-baseweb="slider"] > div > div > div:first-child { background-color: #FF5E94 !important; }
</style>
""",
    unsafe_allow_html=True,
)


def card(k: str, v: str):
    st.markdown(f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div></div>', unsafe_allow_html=True)


STEP_MAIN, STEP_SIMULATOR, STEP_CLUSTER = "main", "simulator", "cluster"

if "step" not in st.session_state:
    st.session_state.step = STEP_MAIN


def go_to_cluster_page(cluster_idx):
    st.session_state.target_cluster_idx = cluster_idx
    st.session_state.step = STEP_CLUSTER


def go(step: str):
    st.session_state.step = step
    st.rerun()


# ---------------------------------------------------------
# 3. 데이터 로드 및 전처리
# ---------------------------------------------------------
rfm_features = ['activity_score', 'adjusted_frequency', 'Monetary', 'Engagement', 'subscription_risk',
                'support_pressure']

CLUSTER_COLORS = {
    "Cluster 0": "#4ADE80",  # 맑은 파스텔 그린
    "Cluster 1": "#FBBF24",  # 맑은 파스텔 옐로우오렌지
    "Cluster 2": "#FF6B6B",  # 🚨 맑은 파스텔 레드
    "Cluster 3": "#60A5FA"  # 맑은 파스텔 블루
}


@st.cache_data
def load_data():
    file_path = os.path.join(project_root, '01_data', 'processed', 'model_df.csv')

    if not os.path.exists(file_path):
        st.error(f"🚨 파일을 찾을 수 없습니다: `{file_path}`")
        st.stop()

    rfm_df, rfm_scaled_df = make_rfm_table(file_path)

    model_df = pd.read_csv(file_path)
    if 'churned' in model_df.columns:
        rfm_df['churned'] = model_df['churned']
    else:
        rfm_df['churned'] = 0

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm_df['Cluster'] = [f"Cluster {c}" for c in kmeans.fit_predict(rfm_scaled_df)]

    return rfm_df


@st.cache_resource
def load_model(_df):
    X = _df[rfm_features]
    y = _df['churned']

    # 🌟 XGBoost 모델로 교체 (주피터 노트북의 최적 파라미터 적용) 🌟
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )
    model.fit(X, y)
    return model

df_all = load_data()
model_xgb = load_model(df_all) # 변수명 변경


# ---------------------------------------------------------
# 4. 라우팅 렌더링
# ---------------------------------------------------------
def render_main():
    st.markdown('<div class="hero-title">🎵 음악 구독 고객 이탈 예측 서비스 🎵</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">이탈 위험도 예측 시뮬레이터 및 맞춤형 고객 군집별 대응 전략</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(
            '<div class="landing-card"><div class="title">🤖 이탈 시뮬레이터 (XAI)</div><div class="desc">특정 고객의 행동 지표를 입력하여 이탈 위험도를 예측하고 맞춤형 방어 전략을 도출합니다.</div></div>',
            unsafe_allow_html=True)
        if st.button("시뮬레이터 및 리텐션 전략", type="primary", use_container_width=True): go(STEP_SIMULATOR)
    with c2:
        st.markdown(
            '<div class="landing-card"><div class="title">👥 고객 군집별 대응 전략</div><div class="desc">K-Means로 세분화된 4가지 고객 군집의 이탈률, 행동 분포, 맞춤형 타겟 전략을 확인합니다.</div></div>',
            unsafe_allow_html=True)
        if st.button("고객 군집별 대응 전략", type="primary", use_container_width=True): go(STEP_CLUSTER)


def render_simulator():
    st.title("🤖 이탈 시뮬레이터 및 리텐션 전략")

    with st.sidebar:
        st.header("🎧 행동 지표 입력")
        if st.button("⬅ 메인으로"): go(STEP_MAIN)
        st.divider()
        val_act = st.slider(
            "🏃 활동 점수", float(df_all['activity_score'].min()), float(df_all['activity_score'].max()),
            float(df_all['activity_score'].mean()),
            help="사용자의 알림 클릭 횟수를 기반으로 한 최근 활동 수준 지표입니다."
        )
        val_adj_freq = st.slider(
            "🔄 실질 소비 빈도", float(df_all['adjusted_frequency'].min()), float(df_all['adjusted_frequency'].max()),
            float(df_all['adjusted_frequency'].mean()),
            help="노래 재생 횟수에 스킵 비율을 반영한 실제 콘텐츠 소비 빈도 지표입니다."
        )
        val_monetary = st.slider(
            "💰 청취 시간", float(df_all['Monetary'].min()), float(df_all['Monetary'].max()),
            float(df_all['Monetary'].mean()),
            help="주간 청취 시간을 기반으로 한 서비스 이용 가치 지표입니다."
        )
        val_eng = st.slider(
            "🔥 플랫폼 몰입도", float(df_all['Engagement'].min()), float(df_all['Engagement'].max()),
            float(df_all['Engagement'].mean()),
            help="플레이리스트 생성, 친구 추가, 공유 활동을 합산한 플랫폼 참여도 지표입니다."
        )
        val_sub_risk = st.slider(
            "⚠️ 구독 중단 위험도", float(df_all['subscription_risk'].min()), float(df_all['subscription_risk'].max()),
            float(df_all['subscription_risk'].median()),
            help="구독 일시정지 횟수를 기반으로 한 사용자 이탈 위험 신호 지표입니다."
        )
        val_sup_press = st.slider(
            "📞 고객센터 문의 빈도", float(df_all['support_pressure'].min()), float(df_all['support_pressure'].max()),
            float(df_all['support_pressure'].median()),
            help="고객센터 문의 횟수를 기반으로 한 서비스 불만 또는 문제 경험 지표입니다."
        )
        st.divider()
        run_sim = st.button("▶️ 예측 실행", type="primary")

    if not run_sim:
        st.info("👈 왼쪽 사이드바에서 고객의 행동 지표를 조절한 뒤 **'▶️ 예측 실행'** 버튼을 눌러주세요.")
        return

    input_df = pd.DataFrame(
        {'activity_score': [val_act], 'adjusted_frequency': [val_adj_freq], 'Monetary': [val_monetary],
         'Engagement': [val_eng], 'subscription_risk': [val_sub_risk], 'support_pressure': [val_sup_press]})
    prob = model_xgb.predict_proba(input_df)[0][1] * 100
    pred_label = "🔇 이탈 고위험군" if prob >= 50 else "▶️ 안정(유지) 고객"

    c1, c2, c3 = st.columns([1, 1, 1.5], gap="medium")
    with c1:
        card("예측 결과", pred_label)
    with c2:
        card("이탈 확률", f"{prob:.1f}%")
    with c3:
        if prob >= 50:
            st.error("🚨 **이탈 위험이 매우 높습니다!** 즉각적인 리텐션 액션이 필요합니다.")
        elif prob >= 30:
            st.warning("⚠️ **이탈 징후가 보입니다.** 선제적 타겟 마케팅을 고려하세요.")
        else:
            st.success("✅ **안정적으로 서비스를 이용 중**인 우수 고객입니다.")

    st.divider()

    col_shap_graph, col_shap_desc = st.columns([1.5, 1], gap="large")

    with col_shap_graph:
        st.subheader("💡 XAI 원인 분석")

        plt.style.use('default')
        explainer = shap.TreeExplainer(model_xgb)
        shap_values = explainer.shap_values(input_df)
        fig_shap, ax_shap = plt.subplots(figsize=(6, 4.5))

        shap.decision_plot(explainer.expected_value, shap_values, features=input_df,
                           feature_names=input_df.columns.tolist(), show=False)
        fig_shap.patch.set_facecolor('#F8F9FA')
        ax_shap.set_facecolor('#F8F9FA')
        st.pyplot(fig_shap)

    with col_shap_desc:
        st.markdown("<br><br>", unsafe_allow_html=True)

        shap_vals = shap_values[0]
        feature_names = input_df.columns.tolist()
        shap_dict = dict(zip(feature_names, shap_vals))

        sorted_shap = sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)
        top_churn_factor = sorted_shap[0]
        top_retain_factor = sorted_shap[-1]

        dynamic_desc = f"""
        <div style='background-color: #FFFFFF; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);'>
            <h5 style='margin-top:0; color:#1F2937;'>📊 XAI 동적 해석 가이드</h5>
            <ul style='margin:0; padding-left: 20px; color: #4B5563; font-size: 0.95rem; line-height: 1.8;'>
        """

        if top_churn_factor[1] > 0:
            dynamic_desc += f"<li>현재 이 고객의 <b>이탈 확률을 가장 크게 높이는 요인(붉은색)</b>은 <b style='color:#EF4444;'>{top_churn_factor[0]}</b> 입니다.</li>"
        if top_retain_factor[1] < 0:
            dynamic_desc += f"<li>이 고객의 <b>구독 유지를 돕는 긍정적 요인(푸른색)</b>은 <b style='color:#3B82F6;'>{top_retain_factor[0]}</b> 입니다.</li>"

        dynamic_desc += "<li>가운데 선(평균)에서 우측으로 멀어질수록 위험도가 높다는 뜻입니다.</li></ul></div>"

        st.markdown(dynamic_desc, unsafe_allow_html=True)

    st.divider()

    st.subheader("🎯 타겟 리텐션 전략")

    if val_monetary >= df_all['Monetary'].quantile(0.75):
        target_cluster = "Cluster 2"
        cluster_desc = "핵심 헤비 콘텐츠 소비 사용자"
    elif val_act <= df_all['activity_score'].quantile(0.25) and val_sub_risk >= df_all['subscription_risk'].quantile(
            0.75):
        target_cluster = "Cluster 1"
        cluster_desc = "활동 감소 이탈 위험 사용자"
    elif val_act >= df_all['activity_score'].quantile(0.75) and val_sub_risk >= df_all['subscription_risk'].quantile(
            0.75):
        target_cluster = "Cluster 3"
        cluster_desc = "활동 높지만 잠재적 불만 사용자"
    else:
        target_cluster = "Cluster 0"
        cluster_desc = "안정적 콘텐츠 소비 사용자"

    # 예측된 군집의 고유 색상으로 텍스트 색깔 표시
    target_color = CLUSTER_COLORS[target_cluster]

    st.markdown(
        f"<p style='font-size:1.1rem; color:#1F2937;'>분석 결과, 이 고객은 <b style='color:{target_color};'>`{cluster_desc} ({target_cluster})`</b> 행동 패턴을 보입니다.</p>",
        unsafe_allow_html=True)
    st.markdown("<p style='color:#6B7280;'>해당 고객 군집이 가진 행동 특징과 맞춤형 전략을 자세히 확인해 보세요.</p>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    cluster_idx_num = int(target_cluster[-1])
    st.button("대응 전략 확인해보러 가기 ➡️", type="primary", on_click=go_to_cluster_page, args=(cluster_idx_num,))


def render_cluster_profile():
    st.title("👥 맞춤형 고객 군집별 대응 전략")

    st.info("💡 선택하신 **군집**의 이탈률 위치와 맞춤형 전략을 확인하세요.")

    with st.sidebar:
        st.header("🔍 군집 탐색 설정")
        if st.button("⬅ 메인으로"): go(STEP_MAIN)
        st.divider()
        st.markdown("👇 **아래에서 분석할 군집을 선택하세요.**")

        default_idx = st.session_state.get('target_cluster_idx', 0)
        target_cluster = st.selectbox("🔍 분석할 군집 선택", ["Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3"],
                                      index=default_idx)

    current_color = CLUSTER_COLORS[target_cluster]

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📊 군집별 이탈률 비교")
        churn_rates = df_all.groupby('Cluster')['churned'].mean().reset_index()
        churn_rates['churned'] = churn_rates['churned'] * 100

        bar_colors = [current_color if c == target_cluster else '#E5E7EB' for c in churn_rates['Cluster']]

        fig_bar = px.bar(churn_rates, x='Cluster', y='churned', text_auto='.1f', color='Cluster',
                         color_discrete_sequence=bar_colors)
        fig_bar.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                              yaxis_title="이탈률 (%)", margin=dict(t=30, b=20))
        fig_bar.update_traces(textfont_size=14, textangle=0, textposition="outside", cliponaxis=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        st.subheader(f"🎯 {target_cluster} 행동 분포 (vs 전체)")
        scaler_minmax = MinMaxScaler()
        df_scaled = df_all.copy()
        df_scaled[rfm_features] = scaler_minmax.fit_transform(df_scaled[rfm_features])

        target_avg = df_scaled[df_scaled['Cluster'] == target_cluster][rfm_features].mean().reset_index()
        target_avg.columns = ['Feature', 'Score']

        target_avg['Feature'] = target_avg['Feature'].map({
            'activity_score': '활동 점수', 'adjusted_frequency': '실질 소비 빈도',
            'Monetary': '청취 시간', 'Engagement': '플랫폼 몰입도',
            'subscription_risk': '구독 중단 위험', 'support_pressure': '고객센터 문의'
        })
        target_avg['Type'] = target_cluster

        total_avg = df_scaled[rfm_features].mean().reset_index()
        total_avg.columns = ['Feature', 'Score']
        total_avg['Feature'] = target_avg['Feature']
        total_avg['Type'] = '전체 평균'

        radar_df = pd.concat([target_avg, total_avg])

        fig_radar = px.line_polar(radar_df, r='Score', theta='Feature', color='Type', line_close=True, markers=True,
                                  color_discrete_map={target_cluster: current_color, '전체 평균': '#9CA3AF'})
        fig_radar.update_traces(fill='toself', opacity=0.5)
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 1])), plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=30, b=20, l=40, r=40),
                                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
        st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    feat_names = ['활동 점수', '소비 빈도', '청취 시간', '플랫폼 몰입도', '구독 위험도', '불만(고객센터)']

    if target_cluster == "Cluster 0":
        title = "안정적 콘텐츠 소비 사용자"
        desc = "이 군집은 콘텐츠 소비 빈도가 높은 안정적인 사용자 그룹입니다. 플랫폼 기능 참여도도 비교적 높은 편이며 서비스 이용 패턴이 꾸준하게 유지되는 특징을 보입니다. 구독 일시 중단이나 이탈 위험 신호가 낮아 장기적인 서비스 이용 가능성이 높은 사용자군입니다."
        levels = [2, 3, 2, 3, 1, 2]
        texts = ['중간', '높음', '중간', '높음', '낮음', '보통']
        strategy_msg = "• 개인화 음악 추천 알고리즘 강화<br>• 신규 콘텐츠 및 아티스트 적극 추천<br>• 플레이리스트 추천 큐레이션 기능 개선<br>• 친구 간 소셜 및 콘텐츠 공유 기능 활성화"
        bg_color = "#F0FDF4"  # 연한 그린 배경

    elif target_cluster == "Cluster 1":
        title = "활동 감소 이탈 위험 사용자"
        desc = "이 군집은 서비스 활동 수준이 낮고 구독 일시 중단 경험이 많은 사용자 그룹입니다. 최근 서비스 이용 빈도가 감소하고 있으며 장기적으로 서비스 이탈 가능성이 높은 사용자군으로 해석됩니다. 서비스 이용 경험이 감소하면서 플랫폼에서 점차 멀어지는 패턴을 보입니다."
        levels = [1, 2, 2, 3, 3, 2]
        texts = ['낮음', '중간', '중간', '높음', '높음', '보통']
        strategy_msg = "• 앱 접속을 유도하는 재참여 알림(Push) 발송<br>• 취향 맞춤형 개인화 콘텐츠 추천 리텐션 메일<br>• 구독 유지를 위한 특별 프로모션(할인 쿠폰) 제공<br>• 복귀 사용자 대상 타겟 마케팅 캠페인 집중 전개"
        bg_color = "#FFFBEB"  # 연한 옐로우 배경

    elif target_cluster == "Cluster 2":
        title = "핵심 헤비 콘텐츠 소비 사용자"
        desc = "이 군집은 청취 시간이 가장 높은 핵심 사용자 그룹입니다. 콘텐츠 소비가 매우 활발하며 서비스 이용 가치가 높은 사용자군으로 볼 수 있습니다. 서비스 만족도가 비교적 높으며 이탈 가능성도 낮은 안정적인 핵심 사용자층입니다."
        levels = [3, 3, 4, 2, 1, 1]
        texts = ['높음', '높음', '가장 높음', '중간', '낮음', '낮음']
        strategy_msg = "• 고음질, 무손실 등 프리미엄 서비스 우선 제공<br>• 개인화 플레이리스트 추천 기능 지속 강화<br>• 최상위 VIP 전용 혜택 및 리워드 프로그램 운영<br>• 신규 기능 도입 시 베타 테스트 참여 기회 부여"
        bg_color = "#FFF0F0"  # 🚨 연한 레드 배경

    elif target_cluster == "Cluster 3":
        title = "활동 높지만 잠재적 불만 사용자"
        desc = "이 군집은 서비스 활동 수준과 플랫폼 참여도가 높은 사용자 그룹입니다. 그러나 구독 일시 중단 경험이 비교적 많아 서비스 이용 과정에서 불편이나 불만을 경험했을 가능성이 있습니다. 서비스 이용 빈도는 높지만 사용자 경험 개선이 필요한 잠재적 이탈 위험 사용자군입니다."
        levels = [4, 2, 2, 3, 3, 2]
        texts = ['가장 높음', '중간', '중간', '높음', '높음', '보통']
        strategy_msg = "• 앱 내 사용성 및 전반적인 사용자 경험(UX) 개선<br>• 고객센터 불만사항 선제적 대응 및 품질 향상<br>• 이탈 원인이 될 수 있는 추천 알고리즘 로직 개선<br>• 심층 피드백 수집 및 서비스 업데이트 내역 투명한 안내"
        bg_color = "#F0F6FF"  # 연한 블루 배경

    st.markdown(f"#### 🔎 {title}")

    st.markdown(f"""
    <div style="background-color: {bg_color}; padding: 25px 30px; border-radius: 16px; margin-bottom: 30px; border: none; box-shadow: 0 4px 6px rgba(0,0,0,0.02);">
        <p style='color:#374151; font-size:1.1rem; line-height:1.7; margin:0; font-weight:500;'>{desc}</p>
    </div>
    """, unsafe_allow_html=True)

    col_t1, col_t2 = st.columns([1.2, 1], gap="large")

    with col_t1:
        st.markdown("##### 📈 주요 행동 지표 프로필 (Profile Chart)")

        df_line = pd.DataFrame({'지표': feat_names, '수준': levels, '설명': texts})
        fig_line = px.line(df_line, x='지표', y='수준', text='설명', markers=True)
        fig_line.update_traces(
            textposition="top center", textfont_size=13,
            marker=dict(size=12, color=current_color, line=dict(width=2, color='#FFFFFF')),
            line=dict(color=current_color, width=3)
        )
        fig_line.update_layout(
            yaxis=dict(range=[0, 5], visible=False), xaxis=dict(title=""),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=20, b=10, l=10, r=10), height=280
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with col_t2:
        st.markdown(f"""
        <div style="background-color: {bg_color}; padding: 30px; border-radius: 16px; border: none; box-shadow: 0 4px 6px rgba(0,0,0,0.02); height:100%;">
            <h4 style="color:#111827; margin-top:0; margin-bottom:20px; font-weight:800;">🎯 맞춤형 대응 전략</h4>
            <p style="color:#4B5563; font-size:1.05rem; line-height:2.4; margin:0; font-weight:500;">
                {strategy_msg}
            </p>
        </div>
        """, unsafe_allow_html=True)


if st.session_state.step == STEP_MAIN:
    render_main()
elif st.session_state.step == STEP_SIMULATOR:
    render_simulator()
elif st.session_state.step == STEP_CLUSTER:
    render_cluster_profile()
else:
    go(STEP_MAIN)