import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title='MOKPO AI 상권 분석', page_icon='🏪', layout='wide')
st.title('🏪 MOKPO AI | 목포 상권 분석')
st.warning('⚠️ 교육용 시제품입니다. 아래 데이터는 실제 목포시 통계가 아닌 가상 데이터입니다. 매출 예측과 군집 결과를 실제 창업 판단에 사용하지 마세요.')
st.caption('인공지능 기초 수행평가 · K-means 비지도학습 + Random Forest 지도학습')

AREAS = ['목포 원도심', '평화광장 주변', '북항 주변', '하당 상권', '목포역 주변', '용해동 주변', '옥암동 주변', '남악 인접 생활권']
CATEGORIES = ['음식점', '카페', '소매점']

@st.cache_data
def make_demo_data():
    rng = np.random.default_rng(20261109)
    # 가상 설정값: 실제 지역 특성에 대한 주장이 아님
    area_base = [210, 460, 290, 380, 330, 240, 350, 310]
    records = []
    for a_idx, area in enumerate(AREAS):
        for month in range(1, 13):
            for category in CATEGORIES:
                for sample in range(8):
                    traffic = max(30, area_base[a_idx] + rng.normal(0, 65) + 35*np.sin(month*np.pi/6))
                    youth = float(np.clip(rng.normal(0.38 + (a_idx % 3)*0.055, 0.07), 0.08, 0.85))
                    competitors = int(rng.integers(4, 31))
                    rent = float(np.clip(rng.normal(75 + a_idx*7, 16), 25, 180))
                    cat_effect = {'음식점': 400, '카페': 120, '소매점': 250}[category]
                    # 교육용으로만 사용하는 가상 매출 생성 공식 (만원/월)
                    sales = max(100, 250 + 4.3*traffic + 650*youth - 10*competitors - 2.2*rent + cat_effect + rng.normal(0, 180))
                    records.append({'지역': area, '월': month, '업종': category, '유동인구지수': round(traffic,1), '청년층비율': round(youth,3), '경쟁점포수': competitors, '임대료지수': round(rent,1), '월매출_만원': round(sales,1)})
    return pd.DataFrame(records)

data = make_demo_data()
FEATURES = ['유동인구지수', '청년층비율', '경쟁점포수', '임대료지수']

@st.cache_resource
def train_models(df):
    # 지역별 평균으로 K-means 학습: 8개 지역의 4개 특성
    area_df = df.groupby('지역', as_index=False)[FEATURES].mean()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(area_df[FEATURES])
    cluster_model = KMeans(n_clusters=3, random_state=42, n_init=10)
    area_df['군집'] = cluster_model.fit_predict(scaled)
    # 지도학습: 가상 매출을 학습, 훈련/테스트 분리
    X = pd.get_dummies(df[FEATURES + ['업종', '월']], columns=['업종'], dtype=int)
    y = df['월매출_만원']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = RandomForestRegressor(n_estimators=100, min_samples_leaf=3, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {'MAE': mean_absolute_error(y_test, predictions), 'R2': r2_score(y_test, predictions), '학습건수': len(X_train), '검증건수': len(X_test)}
    return area_df, cluster_model, scaler, model, list(X.columns), metrics

area_df, cluster_model, scaler, regressor, model_columns, metrics = train_models(data)

with st.sidebar:
    st.header('🔎 분석 조건')
    area = st.selectbox('목포 지역 선택', AREAS)
    category = st.selectbox('업종 선택', CATEGORIES)
    month = st.slider('예측 월', 1, 12, 6)
    st.info('지역 이름과 데이터는 시연용입니다. 실제 지역별 특성을 반영하지 않습니다.')

st.subheader(f'📍 {area} · {category} 분석 리포트')
subset = data[(data['지역'] == area) & (data['업종'] == category)]
row = area_df.loc[area_df['지역'] == area].iloc[0]
# 선택 지역의 해당 업종 평균 특성으로 예측값 구성
avg = subset[FEATURES].mean()
entry = {'유동인구지수': avg['유동인구지수'], '청년층비율': avg['청년층비율'], '경쟁점포수': avg['경쟁점포수'], '임대료지수': avg['임대료지수'], '월': month}
for cat in CATEGORIES:
    entry[f'업종_{cat}'] = int(category == cat)
X_one = pd.DataFrame([entry]).reindex(columns=model_columns, fill_value=0)
estimated = float(regressor.predict(X_one)[0])

c1, c2, c3 = st.columns(3)
c1.metric('AI 상권 군집', f"군집 {int(row['군집'])+1}")
c2.metric('예측 월매출 (가상)', f'{estimated:,.0f}만 원')
c3.metric('선택 업종', category)
st.caption('예측 월매출은 가상으로 생성한 매출 데이터로 학습한 결과이며, 실제 점포 매출 예상치가 아닙니다.')

st.subheader('📊 입력 데이터의 특징 (가상)')
chart_df = pd.DataFrame({'지표': FEATURES, '선택 지역 평균': [float(row[f]) for f in FEATURES]})
st.dataframe(chart_df, use_container_width=True, hide_index=True)
st.bar_chart(data.groupby('지역')['유동인구지수'].mean().rename('가상 유동인구지수'))

st.subheader('💡 데이터 기반 경영전략 검토')
if avg['청년층비율'] > data['청년층비율'].median():
    st.write('• **고객층 가설:** 청년층 비율이 상대적으로 높은 조건입니다. SNS 홍보 및 모바일 접근성을 검토할 수 있습니다.')
else:
    st.write('• **고객층 가설:** 청년층 비율이 상대적으로 낮은 조건입니다. 다양한 연령층의 접근성과 지역 커뮤니티 홍보를 검토할 수 있습니다.')
if avg['경쟁점포수'] > data['경쟁점포수'].median():
    st.write('• **경쟁 가설:** 경쟁점포수가 상대적으로 많아 차별화된 상품·서비스 전략을 검토할 수 있습니다.')
else:
    st.write('• **경쟁 가설:** 경쟁점포수가 상대적으로 적어 잠재 수요와 상권 규모를 추가로 확인할 필요가 있습니다.')
st.caption('경영전략 문장은 AI가 생성한 전략이 아니라 분석 변수에 따라 작동하는 명시적 규칙입니다.')

with st.expander('🧠 머신러닝 학습 및 검증 과정 확인'):
    st.markdown('**비지도학습:** 8개 가상 지역의 4개 특성을 표준화한 후 K-means(k=3)로 군집화했습니다.')
    st.dataframe(area_df[['지역','군집'] + FEATURES], use_container_width=True, hide_index=True)
    st.markdown('**지도학습:** 가상 매출 데이터의 75%로 랜덤 포레스트를 학습하고 25%로 평가했습니다.')
    st.metric('테스트 MAE (가상 매출, 만 원)', f"{metrics['MAE']:,.1f}")
    st.metric('테스트 R² (가상 데이터)', f"{metrics['R2']:.3f}")
    st.caption(f"훈련 {metrics['학습건수']}건 / 테스트 {metrics['검증건수']}건. 같은 가상 생성 규칙에서 만들어진 표본을 나누었으므로 실제 목포 데이터에 대한 성능을 뜻하지 않습니다.")
    importance = pd.Series(regressor.feature_importances_, index=model_columns).sort_values(ascending=False)
    st.bar_chart(importance)
    st.download_button('가상 학습 데이터 CSV 내려받기', data.to_csv(index=False).encode('utf-8-sig'), file_name='mokpo_demo_synthetic.csv', mime='text/csv')

st.divider()
st.caption('참고 기사: 전자신문 (2025.10.28.) 핀다 오픈업 프로 출시 · 본 앱은 핀다와 무관한 학생 교육용 독립 시제품입니다.')
