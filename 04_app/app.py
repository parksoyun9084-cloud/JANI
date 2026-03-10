# -*- coding: utf-8 -*-
"""
음악 구독 서비스 이탈 예측 대시보드 (Light & Green Theme)
실제 데이터(model_df.csv) 연동 및 새로운 5개 Feature 가설 기반 CatBoost 모델
실행: streamlit run 04_app/app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import shap
from catboost import CatBoostClassifier
import os

# ---------------------------------------------------------
# 1. 페이지 설정 및 CSS (Light & Green + 커스텀 초록 슬라이더)
# ---------------------------------------------------------
st.set_page_config(page_title="Music Churn Prediction", page_icon="🎵", layout="wide")

st.markdown(
    """
<style>
/* 전체 배경 및 카드 레이아웃 */
.stApp { background: #F8F9FA; color: #1F2937; }
.block-container { padding-top: 2.5rem !important; padding-bottom: 2.0rem !important; max-width: 1200px; }
section[data-testid="stSidebar"] > div { background-color: #FFFFFF; border-right: 1px solid #E5E7EB; }

.hero-title { font-size: 2.3rem; font-weight: 850; letter-spacing: -0.02em; margin: 0 0 0.35rem 0; color: #111827; }
.hero-subtitle { font-size: 1.05rem; color: #6B7280; margin: 0 0 2rem 0; }

.landing-card, .card {
  background-color: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 16px; padding: 20px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.04); transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.landing-card:hover { transform: translateY(-3px); box-shadow: 0 8px 15px rgba(0, 0, 0, 0.08); }
.landing-card .title { font-size: 1.2rem; font-weight: 800; margin: 0 0 0.6rem 0; color: #111827; }
.landing-card .desc { font-size: 0.95rem; color: #6B7280; line-height: 1.5; }

.card { display:flex; flex-direction:column; gap:6px; min-height: 100px; justify-content: center;}
.card .k { font-size: 0.85rem !important; font-weight: 700 !important; color: #6B7280 !important; }
.card .v { font-size: 1.7rem !important; font-weight: 800 !important; color: #111827 !important; line-height: 1.1; }
.card .s { font-size: 0.9rem !important; font-weight: 700 !important; color: #10B981 !important; } 

/* 기본 버튼 스타일 */
div.stButton > button {
  border-radius: 12px; padding: 0.6rem 1.5rem; font-weight: 700; border: 1px solid #D1D5DB;
  background-color: #FFFFFF; color: #374151; transition: all 0.2s ease;
}
div.stButton > button:hover { background-color: #F3F4F6; border-color: #9CA3AF; }

/* Primary 버튼 (초록색 그라데이션) */
div.stButton > button[kind="primary"] {
  border: none !important; background: linear-gradient(135deg, #10B981, #059669) !important; color: #FFFFFF !important;
  box-shadow: 0 4px 10px rgba(16, 185, 129, 0.3);
}
div.stButton > button[kind="primary"]:hover { transform: scale(1.02); filter: brightness(1.05); }

/* ★ 스트림릿 기본 UI 초록색으로 덮어쓰기 ★ */
.stSlider div[data-baseweb="slider"] div[role="slider"] {
    background-color: #10B981 !important; border: 2px solid #FFFFFF !important; box-shadow: 0 2px 6px rgba(16, 185, 129, 0.4) !important;
}
.stSlider div[data-baseweb="slider"] div[role="slider"]:focus {
    box-shadow: 0 0 0 0.2rem rgba(16, 185, 129, 0.25) !important; outline: none !important;
}
.stSlider div[data-baseweb="slider"] > div > div > div:first-child { background-color: #10B981 !important; }
.stNumberInput input:focus, .stTextInput input:focus, .stSelectbox > div > div:focus-within { border-color: #10B981 !important; box-shadow: 0 0 0 1px #10B981 !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 2. 공통 UI 로직
# ---------------------------------------------------------
def card(k: str, v: str, s: str = ""):
    st.markdown(f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{s}</div></div>',
                unsafe_allow_html=True)


STEP_MAIN, STEP_EDA, STEP_SIMULATOR = "main", "eda", "simulator"

if "step" not in st.session_state:
    st.session_state.step = STEP_MAIN


def go(step: str):
    st.session_state.step = step
    st.rerun()


# ---------------------------------------------------------
# 3. 실제 전처리 데이터 로드 및 모델 학습 (캐싱)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    # 현재 파일(app.py)이 있는 폴더 경로 (04_app)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 04_app에서 한 칸 위(..)로 간 다음, 01_data/processed/model_df.csv 찾기 (절대 경로로 변환)
    file_path = os.path.abspath(os.path.join(current_dir, '..', '01_data', 'processed', 'model_df.csv'))

    # 파일 존재 여부 방어 코드
    if not os.path.exists(file_path):
        st.error(f"🚨 파일을 찾을 수 없습니다: `{file_path}`\n\n지정된 경로에 `model_df.csv` 파일이 있는지 확인해 주세요.")
        st.stop()

    df = pd.read_csv(file_path)

    # 결측치 방어
    df = df.fillna(0)
    return df


@st.cache_resource
def load_model(_df):
    """
    선정된 5개의 변수만 추출해 CatBoost를 실시간 학습시킵니다.
    """
    # 2. 가설 검증을 통해 선정된 변수 5개로 수정
    features = [
        'num_subscription_pauses',
        'song_skip_rate',
        'weekly_hours',
        'engagement_score',
        'customer_service_inquiries'
    ]

    # 안전 장치: 컬럼이 없으면 0(또는 임의값)으로 생성 (앱 다운 방지)
    for col in features:
        if col not in _df.columns:
            if col == 'customer_service_inquiries':
                _df[col] = 'Low'
            else:
                _df[col] = 0

    X = _df[features]
    y = _df['churned'] if 'churned' in _df.columns else np.random.choice([0, 1], size=len(_df))  # 타겟 변수 방어

    # customer_service_inquiries 가 문자열(Low, Mid, High)일 경우 CatBoost가 인식할 수 있도록 처리
    cat_features = [col for col in X.columns if X[col].dtype == 'object']

    # 모델 정의 및 학습
    model = CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=5,
        cat_features=cat_features,
        verbose=False,
        random_seed=42,
        auto_class_weights='Balanced'
    )
    model.fit(X, y)
    return model


# 데이터 및 모델 불러오기 실행
df_all = load_data()
model_cb = load_model(df_all)


# ---------------------------------------------------------
# 4. 각 페이지 렌더링 함수
# ---------------------------------------------------------
def render_main():
    st.markdown('<div class="hero-title">🎵 음악 구독 서비스 고객 이탈 예측 서비스</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">전체 고객 데이터 탐색(EDA) 또는 개별 고객 이탈 위험도(XAI) 분석을 선택하세요.', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown(
            '''
            <div class="landing-card">
              <div class="title">📊 전체 데이터 탐색 (EDA)</div>
              <div class="desc">현재 구독자들의 활동 패턴과 이탈자들의 주요 징후를 시각적으로 탐색합니다.</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )
        if st.button("탐색적 데이터 분석(EDA)", type="primary", use_container_width=True):
            go(STEP_EDA)

    with c2:
        st.markdown(
            '''
            <div class="landing-card">
              <div class="title">🤖 고객 이탈 시뮬레이터 (XAI)</div>
              <div class="desc">특정 고객의 행동 데이터를 바탕으로 이탈 위험도를 예측하고 맞춤 전략을 확인합니다.</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )
        if st.button("시뮬레이터 및 리텐션 전략", type="primary", use_container_width=True):
            go(STEP_SIMULATOR)


def render_eda():
    st.title("📊 데이터 탐색 및 핵심 지표 (EDA)")

    with st.sidebar:
        st.header("탐색 설정")
        if st.button("⬅ 메인으로", key="back_to_main_from_eda"):
            go(STEP_MAIN)

        st.divider()
        # 타겟을 제외한 컬럼만 EDA
        features_to_plot = df_all.columns.drop('churned') if 'churned' in df_all.columns else df_all.columns
        target_feat = st.selectbox("분석할 행동 변수 선택", features_to_plot)

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        card("총 분석 데이터 수", f"{len(df_all):,}건", "Processed Data")
    with c2:
        churn_rate = df_all['churned'].mean() * 100 if 'churned' in df_all.columns else 0
        card("전체 이탈률", f"{churn_rate:.1f}%", "Churn Rate")
    with c3:
        avg_eng = df_all['engagement_score'].mean() if 'engagement_score' in df_all.columns else 0
        card("평균 몰입도", f"{avg_eng:,.0f}점", "Engagement Score")
    with c4:
        avg_skip = df_all['song_skip_rate'].mean() if 'song_skip_rate' in df_all.columns else 0
        card("평균 스킵 비율", f"{avg_skip:.2f}", "Song Skip Rate")

    st.divider()

    if 'churned' in df_all.columns:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("💿 활성 vs 이탈 사용자 비율")
            pie_data = df_all['churned'].value_counts().reset_index()
            pie_data.columns = ['Churned', 'Count']
            pie_data['Churned'] = pie_data['Churned'].map({0: '유지 (Active)', 1: '이탈 (Churned)'})

            fig1 = px.pie(pie_data, values='Count', names='Churned', hole=0.5,
                          color='Churned', color_discrete_map={'유지 (Active)': '#10B981', '이탈 (Churned)': '#9CA3AF'})
            fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.subheader(f"🎶 [{target_feat}] 변수 분포 비교")
            fig2 = px.histogram(df_all, x=target_feat, color='churned', barmode='overlay',
                                nbins=30, opacity=0.75,
                                color_discrete_map={0: '#10B981', 1: '#6B7280'})
            fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                               legend_title_text='이탈 여부 (0:유지, 1:이탈)')
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("데이터에 'churned' (이탈 여부) 컬럼이 존재하지 않아 시각화를 제공할 수 없습니다.")


def render_simulator():
    st.title("🤖 이탈 시뮬레이터 및 리텐션 전략")

    with st.sidebar:
        st.header("🎧 핵심 고객 행동 데이터 입력")
        if st.button("⬅ 메인으로", key="back_to_main_from_sim"):
            go(STEP_MAIN)

        st.divider()
        # 3. 요청하신 새 변수 목록 및 수치 제약에 맞춰 UI 업데이트
        val_hours = st.slider("💿 주당 청취 시간 (시간)", 0.0, 50.0, 25.0, help="14시간 이하 시 이탈률 상승")
        val_pauses = st.slider("⏸️ 구독 일시정지 횟수", 0, 4, 1, help="3회 이상부터 이탈 확률 급증")
        val_skip = st.slider("⏭️ 노래 스킵 비율", 0.0, 1.0, 0.5, help="0.6 이상부터 이탈 확률 급증")
        val_eng = st.slider("🔥 몰입도 점수 (Engagement)", 0, 15000, 3000, step=100, help="1500 미만일 경우 이탈 위험 구간")

        # 'Low, Mid, High'로 선택할 수 있는 Selectbox 추가
        val_cs = st.selectbox("📞 고객센터 문의 빈도", ["Low", "Mid", "High"], index=0, help="High일 경우 강한 이탈 신호")

        st.divider()
        run_sim = st.button("▶️ 예측 실행", type="primary", key="run_sim")

    if not run_sim:
        st.info("👈 왼쪽 사이드바에서 유저의 행동 지표를 조절한 뒤 **'▶️ 예측 실행'** 버튼을 눌러주세요.")
        return

    # 모델에 넣을 DataFrame 생성 (순서와 이름은 모델 학습 때와 동일해야 함)
    input_df = pd.DataFrame({
        'num_subscription_pauses': [val_pauses],
        'song_skip_rate': [val_skip],
        'weekly_hours': [val_hours],
        'engagement_score': [val_eng],
        'customer_service_inquiries': [val_cs]
    })

    # 예측 수행
    prob = model_cb.predict_proba(input_df)[0][1] * 100
    pred_label = "🔇 이탈 고위험군" if prob >= 50 else "▶️ 안정(유지) 고객"

    c1, c2, c3 = st.columns([1, 1, 1.5], gap="medium")
    with c1:
        card("예측 결과", pred_label, "CatBoost ML 판정")
    with c2:
        card("이탈 확률", f"{prob:.1f}%", "Probability")
    with c3:
        if prob >= 50:
            st.error("🚨 **이탈 위험이 매우 높습니다!** 즉각적인 리텐션 액션이 필요합니다.")
        elif prob >= 30:
            st.warning("⚠️ **이탈 징후가 보입니다.** 선제적 타겟 마케팅을 고려하세요.")
        else:
            st.success("✅ **안정적으로 서비스를 이용 중**인 우수 고객입니다.")

    st.divider()

    st.subheader("💡 왜 이런 예측이 나왔을까요? (XAI 원인 분석)")

    col_shap, col_action = st.columns([1.2, 1])

    with col_shap:
        st.markdown("<span style='color:#6B7280; font-size:0.9em;'>※ 붉은색: 해지 확률 증가 요인 / 푸른색: 유지 확률 증가 요인</span>",
                    unsafe_allow_html=True)
        explainer = shap.TreeExplainer(model_cb)
        shap_values = explainer.shap_values(input_df)
        expected_val = explainer.expected_value

        fig_shap, ax_shap = plt.subplots(figsize=(6, 4))
        plt.style.use('default')
        shap.decision_plot(expected_val, shap_values, features=input_df, feature_names=input_df.columns.tolist(),
                           show=False)
        fig_shap.patch.set_facecolor('#F8F9FA')
        ax_shap.set_facecolor('#F8F9FA')
        st.pyplot(fig_shap)

    with col_action:
        st.markdown("### 🎯 타겟 리텐션(Retention) 전략")

        shap_abs = np.abs(shap_values[0])
        top_idx = np.argmax(shap_abs)
        top_feature = input_df.columns[top_idx]

        st.markdown(f"현재 고객의 예측에 가장 큰 영향을 미친 지표는 **`{top_feature}`** 입니다.")

        # 4. 새로 변경된 변수에 맞춘 리텐션 전략 로직
        html_box = '<div style="background-color:#FFFFFF; padding:20px; border-radius:12px; border: 1px solid #E5E7EB; border-left: 6px solid #10B981; box-shadow: 0 2px 4px rgba(0,0,0,0.02); color:#1F2937;">'

        if prob >= 50:
            if top_feature == 'customer_service_inquiries':
                html_box += "<b>[서비스 불만 해소]</b><br>고객센터 문의 빈도가 높습니다(High). 다음 달 구독료 30% 할인 쿠폰을 포함한 사과 및 VIP 케어 메일을 즉시 발송하세요."
            elif top_feature == 'num_subscription_pauses':
                html_box += "<b>[구독 유지 불안정 해소]</b><br>구독 일시정지 횟수가 잦아 완전 해지 가능성이 높습니다. 서비스 재개 시 보너스 포인트나 제휴 할인 혜택을 제공하여 이탈을 방지하세요."
            elif top_feature == 'song_skip_rate':
                html_box += "<b>[음악 권태기 극복]</b><br>곡 스킵 비율이 위험 수치입니다. 유저 취향에 맞는 <b>'새로운 발견'</b> 믹스 플레이리스트를 앱 푸시로 전송하여 흥미를 다시 유발하세요."
            elif top_feature == 'engagement_score' or top_feature == 'weekly_hours':
                html_box += "<b>[몰입도 증대 및 관심 환기]</b><br>주당 청취 시간이 저조하거나 전반적인 몰입도가 떨어졌습니다. <b>'이번 주 인기 차트'</b> 등 가볍게 들을 수 있는 플레이리스트 알림을 발송하세요."
            else:
                html_box += "<b>[단기 이탈 방어]</b><br>활동이 저조합니다. 이번 주말 한정 프리미엄 혜택 알림을 보내 접속을 유도하세요."
        else:
            if val_eng >= 5000 and val_hours >= 15:
                html_box += "<b>[VIP 우대]</b><br>몰입도와 청취 시간이 훌륭한 최고의 충성 고객입니다! 연간(Yearly) 플랜 전환 시 특별 할인 혜택을 제공하여 장기 락인(Lock-in) 하세요."
            else:
                html_box += "<b>[이용 활성화 장려]</b><br>구독을 안정적으로 유지하고 있습니다. 앞으로 더 다양한 곡을 소비하도록 신곡 알림이나 에디터 추천 플레이리스트를 노출해 보세요."

        html_box += "</div>"

        st.markdown(html_box, unsafe_allow_html=True)


# ---------------------------------------------------------
# 5. 라우터 실행
# ---------------------------------------------------------
if st.session_state.step == STEP_MAIN:
    render_main()
elif st.session_state.step == STEP_EDA:
    render_eda()
elif st.session_state.step == STEP_SIMULATOR:
    render_simulator()
else:
    st.session_state.step = STEP_MAIN
    st.rerun()