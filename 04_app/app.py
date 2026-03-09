# -*- coding: utf-8 -*-
"""
음악 구독 서비스 이탈 예측 대시보드 (Light & Green Theme)
실제 데이터(data/train.csv) 연동 및 CatBoost 실시간 학습 적용
실행: streamlit run app.py
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
.stNumberInput input:focus, .stTextInput input:focus { border-color: #10B981 !important; box-shadow: 0 0 0 1px #10B981 !important; }
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
# 3. 실제 데이터 로드 및 전처리 (캐싱)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    file_path = 'data/train.csv'

    # 1. 파일 존재 여부 확인
    if not os.path.exists(file_path):
        st.error(f"🚨 파일을 찾을 수 없습니다: `{file_path}`\n\n프로젝트 폴더 안에 `data` 폴더를 만들고 `train.csv` 파일을 넣어주세요.")
        st.stop()

    df = pd.read_csv(file_path)

    # 2. 전처리 파생 변수 방어 코드 (만약 팀원이 파생변수를 안 만들고 원본을 줬을 경우를 대비)
    if 'social_activity' not in df.columns:
        if 'num_platform_friends' in df.columns and 'num_shared_playlists' in df.columns:
            df['social_activity'] = df['num_platform_friends'] + df['num_shared_playlists']
        else:
            df['social_activity'] = 5  # 임시 대체값

    if 'engagement_score' not in df.columns:
        if 'weekly_unique_songs' in df.columns and 'weekly_hours' in df.columns:
            df['engagement_score'] = df['weekly_hours'] * df['weekly_unique_songs']
        else:
            df['engagement_score'] = df.get('weekly_hours', 20) * 10

    if 'tenure_days' not in df.columns:
        if 'signup_date' in df.columns:
            df['signup_date'] = pd.to_datetime(df['signup_date'])
            ref_date = df['signup_date'].max() + pd.Timedelta(days=1)
            df['tenure_days'] = (ref_date - df['signup_date']).dt.days
        else:
            df['tenure_days'] = 180

    # 3. customer_service_inquiries 가 Low/Medium/High 같은 문자열이라면 숫자로 강제 변환
    if 'customer_service_inquiries' in df.columns and df['customer_service_inquiries'].dtype == 'object':
        mapping = {'Low': 1, 'Medium': 3, 'High': 5}
        df['customer_service_inquiries'] = df['customer_service_inquiries'].map(mapping).fillna(1)

    # 결측치 0으로 채우기 (모델 에러 방지용)
    df = df.fillna(0)
    return df


@st.cache_resource
def load_model(_df):
    """
    실제 데이터를 받아서 UI 시뮬레이터에서 사용할 6개 변수만 추출해 CatBoost를 실시간 학습시킵니다.
    """
    # UI에서 사용하는 변수 목록
    features = ['weekly_hours', 'song_skip_rate', 'customer_service_inquiries',
                'tenure_days', 'social_activity', 'engagement_score']

    # 안전 장치: 컬럼이 없으면 0으로 생성
    for col in features:
        if col not in _df.columns:
            _df[col] = 0

    X = _df[features]
    y = _df['churned']

    # 모델 정의 및 학습 (클래스 불균형 해결 옵션 추가)
    model = CatBoostClassifier(
        iterations=200,  # 학습 속도를 위해 200번만
        learning_rate=0.05,
        depth=5,
        verbose=False,
        random_seed=42,
        auto_class_weights='Balanced'  # 유지 고객이 많은 불균형을 알아서 맞춰줌!
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
    st.markdown('<div class="hero-subtitle">전체 고객 데이터 탐색(EDA) 또는 개별 고객 이탈 위험도(XAI) 분석을 선택하세요.</div>', unsafe_allow_html=True)

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
        features = df_all.columns.drop('churned')
        target_feat = st.selectbox("분석할 행동 변수 선택", features)

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        card("총 리스너 수", f"{len(df_all):,}명", "Active Users")
    with c2:
        churn_rate = df_all['churned'].mean() * 100
        card("전체 구독 해지율", f"{churn_rate:.1f}%", "Churn Rate")
    with c3:
        avg_eng = df_all['engagement_score'].mean()
        card("평균 몰입도", f"{avg_eng:,.0f}점", "Engagement Score")
    with c4:
        avg_tenure = df_all['tenure_days'].mean()
        card("평균 구독 기간", f"{avg_tenure:.0f}일", "Tenure Days")

    st.divider()

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


def render_simulator():
    st.title("🤖 이탈 시뮬레이터 및 리텐션 전략")

    with st.sidebar:
        st.header("🎧 고객 행동 데이터 입력")
        if st.button("⬅ 메인으로", key="back_to_main_from_sim"):
            go(STEP_MAIN)

        st.divider()
        val_hours = st.slider("💿 주당 청취 시간 (시간)", 0.0, 49.0, 20.0)
        val_skip = st.slider("⏭️ 노래 스킵 비율", 0.0, 1.0, 0.5)
        val_cs = st.number_input("📞 고객센터 문의 횟수", 0, 20, 1)
        val_tenure = st.number_input("🗓️ 가입 후 경과 일수", 1, 1500, 180)
        val_social = st.slider("👥 플랫폼 내 친구 수", 0, 50, 5)
        val_eng = st.slider("🔥 몰입도 점수 (Engagement)", 0, 1500, 500)

        st.divider()
        run_sim = st.button("▶️ 예측 실행", type="primary", key="run_sim")

    if not run_sim:
        st.info("👈 왼쪽 사이드바에서 유저의 행동 수치를 조절한 뒤 **'▶️ 예측 실행'** 버튼을 눌러주세요.")
        return

    input_df = pd.DataFrame({
        'weekly_hours': [val_hours],
        'song_skip_rate': [val_skip],
        'customer_service_inquiries': [val_cs],
        'tenure_days': [val_tenure],
        'social_activity': [val_social],
        'engagement_score': [val_eng]
    })

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

        # 박스 디자인과 글자를 하나의 HTML 문자열로 합치기
        html_box = '<div style="background-color:#FFFFFF; padding:20px; border-radius:12px; border: 1px solid #E5E7EB; border-left: 6px solid #10B981; box-shadow: 0 2px 4px rgba(0,0,0,0.02); color:#1F2937;">'

        if prob >= 50:
            if top_feature == 'customer_service_inquiries':
                html_box += "<b>[서비스 불만 해소]</b><br>최근 고객센터 문의가 잦습니다. 다음 달 구독료 30% 할인 쿠폰을 포함한 VIP 케어 메일을 즉시 발송하세요."
            elif top_feature == 'song_skip_rate' or top_feature == 'engagement_score':
                html_box += "<b>[음악 권태기 극복]</b><br>곡 스킵 비율이 높습니다. 유저의 취향에 맞는 <b>'새로운 발견'</b> 믹스 플레이리스트를 앱 푸시로 전송하여 흥미를 유발하세요."
            else:
                html_box += "<b>[관심 환기]</b><br>앱 접속 유도를 위해 맞춤형 <b>'Weekly 인기 차트'</b> 알림을 발송하세요."
        else:
            if val_social < 5:
                html_box += "<b>[소셜 락인 강화]</b><br>충성도는 높으나 소셜 기능 활용도가 낮습니다. <b>'친구 초대 시 리워드 지급'</b> 프로모션을 노출하여 이탈 장벽을 높이세요."
            else:
                html_box += "<b>[VIP 우대]</b><br>최고의 충성 고객입니다! 연간(Yearly) 플랜 전환 시 혜택을 제공하여 장기 고객으로 굳히세요."

        html_box += "</div>"

        # 합친 내용을 한 번에 출력
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