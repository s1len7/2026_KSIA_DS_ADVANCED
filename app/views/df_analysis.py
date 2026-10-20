import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import (HistGradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_predict, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from common import (BLUE, GRAY, INK, MUTED, ORANGE, RED, caveat, load_csv, page_header, show,
                    show_describe, style_fig)

DROP = ['Target', 'Defect', 'Flux840s', 'Temp_Oxid', 'Vapor', 'Wavelength', '두께정상', 'CD정상']


@st.cache_data
def get_data():
    df = load_csv('df_analysis.csv', encoding='cp949')
    X = df.drop(columns=DROP).copy()
    X['type'] = (X['type'] == 'wet').astype(int)
    X = pd.get_dummies(X, columns=['UV_type'], dtype=int).astype(float)
    return df, X


@st.cache_data(show_spinner='모델 교차검증 중... (최초 1회)')
def model_scores(_X, y, yd):
    kf = KFold(5, shuffle=True, random_state=0)
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    reg = {
        'Ridge': make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))),
        'RandomForest': RandomForestRegressor(300, min_samples_leaf=3, n_jobs=-1, random_state=0),
        'HistGB': HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0),
    }
    clf = {
        'Logistic': make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight='balanced', max_iter=2000)),
        'RandomForest': RandomForestClassifier(300, min_samples_leaf=3, class_weight='balanced', n_jobs=-1, random_state=0),
    }
    reg_rows = []
    for name, m in reg.items():
        r2 = cross_val_score(m, _X, y, cv=kf, scoring='r2')
        mae = -cross_val_score(m, _X, y, cv=kf, scoring='neg_mean_absolute_error')
        reg_rows.append({'모델': name, 'R²': r2.mean(), 'R² 표준편차': r2.std(), 'MAE': mae.mean()})
    clf_rows = []
    for name, m in clf.items():
        p = cross_val_predict(m, _X, yd, cv=skf, method='predict_proba')[:, 1]
        clf_rows.append({'모델': name, 'ROC-AUC': roc_auc_score(yd, p), 'PR-AUC': average_precision_score(yd, p)})
    return pd.DataFrame(reg_rows), pd.DataFrame(clf_rows)


@st.cache_data(show_spinner='변수 중요도 계산 중...')
def importance(_X, y):
    Xtr, Xte, ytr, yte = train_test_split(_X, y, test_size=0.25, random_state=0)
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(Xtr, ytr)
    pi = permutation_importance(m, Xte, yte, n_repeats=10, random_state=0, scoring='r2')
    return pd.Series(pi.importances_mean, _X.columns).sort_values(ascending=False)


@st.cache_data
def tree_rules(_X, yd, depth):
    t = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=20, random_state=0).fit(_X, yd)
    return export_text(t, feature_names=list(_X.columns), show_weights=True)


df, X = get_data()
y, yd = df['Target'], df['Defect']

page_header('df_analysis — 반도체 공정 불량 요인', '웨이퍼별 산화 → 포토 → 식각 → 이온주입 → 열처리 공정 조건과 품질 결과',
            'df_analysis.csv', 'df_analysis.ipynb')

reg_df, clf_df = model_scores(X, y, yd)
k = st.columns(5)
k[0].metric('웨이퍼 수', f'{len(df):,}')
k[1].metric('불량률 (Defect)', f'{yd.mean():.1%}', f'{yd.sum()}개', delta_color='off')
k[2].metric('Target 평균', f'{y.mean():.1f}', f'중앙값 {y.median():.0f}', delta_color='off')
k[3].metric('Target 회귀 R²', f'{reg_df["R²"].max():.3f}', 'HistGB · 5-fold', delta_color='off')
k[4].metric('Defect 분류 AUC', f'{clf_df["ROC-AUC"].max():.3f}', 'RandomForest', delta_color='off')

tabs = st.tabs(['데이터 개요', '라벨 구조', '영향 인자', '모델 성능', '규칙·고위험 조건'])

# ── 데이터 개요 ──────────────────────────────────────────
with tabs[0]:
    st.markdown(f'**{len(df):,}행 × {df.shape[1]}열**, 결측치 {df.isna().sum().sum()}개, 중복 행 {df.duplicated().sum()}개')
    st.dataframe(df.head(20), width='stretch', height=280)
    st.markdown('##### 기술통계')
    show_describe(df.select_dtypes('number'))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('##### 제거한 컬럼')
        st.dataframe(pd.DataFrame({
            '컬럼': ['Flux840s', 'Temp_Oxid', 'Vapor', 'Wavelength'],
            '이유': ['상수', '상수', '`type` 과 1:1 (dry=O2, wet=H2O)', '`UV_type` 과 1:1 (G=436, H=405, I=365)'],
        }), hide_index=True, width='stretch')
    with c2:
        st.markdown('##### 범주형 변수별 불량률')
        cat = st.selectbox('변수', ['type', 'UV_type', '두께정상', 'CD정상'])
        g = df.groupby(cat)['Defect'].agg(불량률='mean', n='size').reset_index()
        fig = px.bar(g, x=cat, y='불량률', text=g['불량률'].map(lambda v: f'{v:.1%}'),
                     hover_data={'n': ':,'}, color_discrete_sequence=[BLUE])
        fig.update_yaxes(tickformat='.0%')
        show(style_fig(fig, 300, legend=False))

# ── 라벨 구조 ────────────────────────────────────────────
with tabs[1]:
    st.markdown('세 품질 라벨은 모두 **다른 컬럼에 임계값을 적용해 만든 파생값**입니다.')
    st.dataframe(pd.DataFrame({
        '라벨': ['Defect', '두께정상', 'CD정상'],
        '규칙': ['Target ≥ 195 → 1', 'thickness ≥ 700 → 정상', '22 ≤ Line_CD ≤ 55 → 정상'],
        '비정상 비율': [f'{yd.mean():.1%}', f'{(df["두께정상"] == "비정상").mean():.1%}', f'{(df["CD정상"] == "비정상").mean():.1%}'],
    }), hide_index=True, width='stretch')
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x='Target', color=yd.map({0: '정상', 1: '불량'}), nbins=50,
                           color_discrete_map={'정상': BLUE, '불량': RED}, labels={'color': ''})
        fig.add_vline(x=194, line_dash='dash', line_color=INK, annotation_text='Target ≥ 195 → Defect')
        fig.update_layout(title='Target 분포', yaxis_title='웨이퍼 수')
        show(style_fig(fig))
    with c2:
        fig = px.scatter(df, x='thickness', y='Line_CD', color='두께정상', symbol='CD정상', opacity=0.7,
                         color_discrete_map={'정상': BLUE, '비정상': ORANGE})
        fig.add_vline(x=700, line_dash='dash', line_color=INK)
        for v in (22, 55):
            fig.add_hline(y=v, line_dash='dot', line_color=MUTED)
        fig.update_layout(title='thickness · Line_CD 임계값')
        show(style_fig(fig))
    caveat('Defect 를 예측할 때 Target 을 입력에 넣으면 정답이 새어 들어갑니다 (leakage). 이후 모든 분류 모델은 Target 을 제외했습니다.')

# ── 영향 인자 ────────────────────────────────────────────
with tabs[2]:
    imp = importance(X, y)
    c1, c2 = st.columns([2, 3])
    with c1:
        top = imp.head(12)[::-1]
        fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation='h',
                               marker_color=[BLUE if v > 0.05 else GRAY for v in top.values],
                               text=[f'{v:.3f}' for v in top.values], textposition='outside'))
        fig.update_layout(title='Target 예측 변수 중요도 (Permutation ΔR²)')
        show(style_fig(fig, 460, legend=False))
    with c2:
        num_cols = [c for c in X.columns if not c.startswith('UV_type') and c != 'type']
        var = st.selectbox('구간별 불량률을 볼 변수', num_cols, index=num_cols.index('Thin F2'))
        q = st.slider('구간 수 (분위)', 3, 10, 5)
        bins = pd.qcut(df[var], q, duplicates='drop')
        g = df.groupby(bins, observed=True).agg(불량률=('Defect', 'mean'), Target평균=('Target', 'mean'), n=('Defect', 'size'))
        g.index = g.index.astype(str)
        fig = go.Figure(go.Bar(x=g.index, y=g['불량률'], marker_color=BLUE,
                               text=g['불량률'].map(lambda v: f'{v:.1%}'), textposition='outside',
                               customdata=g[['Target평균', 'n']],
                               hovertemplate='%{x}<br>불량률 %{y:.1%}<br>Target 평균 %{customdata[0]:.1f}<br>n=%{customdata[1]}<extra></extra>'))
        fig.add_hline(y=yd.mean(), line_dash='dot', line_color=INK, annotation_text=f'전체 {yd.mean():.1%}')
        fig.update_layout(title=f'{var} 구간별 불량률', yaxis_tickformat='.0%')
        show(style_fig(fig, 400, legend=False))

# ── 모델 성능 ────────────────────────────────────────────
with tabs[3]:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('##### Target 회귀 (5-fold CV)')
        st.dataframe(reg_df.style.format({'R²': '{:.3f}', 'R² 표준편차': '{:.3f}', 'MAE': '{:.1f}'}),
                     hide_index=True, width='stretch')
    with c2:
        st.markdown('##### Defect 분류 (Target 제외, 5-fold CV)')
        st.dataframe(clf_df.style.format({'ROC-AUC': '{:.3f}', 'PR-AUC': '{:.3f}'}),
                     hide_index=True, width='stretch')
        st.caption(f'PR-AUC 기준선(불량률) = {yd.mean():.3f}')
    caveat('트리 모델이 선형 모델보다 크게 좋다는 것은 변수 효과가 직선이 아니라 임계값 형태라는 뜻입니다.')

# ── 규칙·고위험 조건 ─────────────────────────────────────
with tabs[4]:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown('##### 조건 시뮬레이터')
        s1, s2, s3 = st.columns(3)
        f4 = s1.slider('Thin F4 >', int(df['Thin F4'].min()), int(df['Thin F4'].max()), 458)
        tox = s2.slider('산화온도(Temp_OXid) <', int(df['Temp_OXid'].min()), int(df['Temp_OXid'].max()) + 1, 900)
        ppm = s3.slider('ppm ≤', float(df['ppm'].min()), float(df['ppm'].max()), float(df['ppm'].max()))
        mask = (df['Thin F4'] > f4) & (df['Temp_OXid'] < tox) & (df['ppm'] <= ppm)
        sub = df[mask]
        m1, m2, m3 = st.columns(3)
        m1.metric('해당 웨이퍼', f'{mask.sum():,}개', f'{mask.mean():.1%}', delta_color='off')
        m2.metric('불량률 (조건 안)', f'{sub["Defect"].mean():.1%}' if len(sub) else '-')
        m3.metric('불량률 (조건 밖)', f'{df.loc[~mask, "Defect"].mean():.1%}' if (~mask).any() else '-')
        if 0 < len(sub) < 30:
            st.warning(f'표본이 {len(sub)}개로 적어 불량률 해석에 주의가 필요합니다.', icon=':material/warning:')

        lowT, f4hi = df['Temp_OXid'] < 900, df['Thin F4'] > 458
        seg = pd.Series(np.select(
            [lowT & f4hi, f4hi & (df['ppm'] <= 28.45), f4hi, lowT],
            ['산화온도<900 & F4>458', 'F4>458 & ppm≤28.45', 'F4>458 (그 외)', '산화온도<900 (F4≤458)'], '기타 (기준)'))
        g = df.groupby(seg)['Defect'].agg(['mean', 'size']).sort_values('mean')
        fig = go.Figure(go.Bar(x=g['mean'], y=g.index, orientation='h',
                               marker_color=[RED if m > .5 else (BLUE if m > .1 else GRAY) for m in g['mean']],
                               text=[f'{m:.0%} (n={n})' for m, n in zip(g['mean'], g['size'])], textposition='outside'))
        fig.update_layout(title='고위험 조건 조합 (위에서부터 우선 배정, 중복 없음)', xaxis_tickformat='.0%', xaxis_range=[0, 1.2])
        show(style_fig(fig, 340, legend=False))
    with c2:
        st.markdown('##### Decision Tree 규칙')
        depth = st.radio('깊이', [2, 3, 4], index=1, horizontal=True)
        st.code(tree_rules(X, yd, depth), language=None)
        st.caption('weights = [정상, 불량] 웨이퍼 수')
    caveat('고위험 조건의 표본이 15~24개로 적습니다. 관측 데이터의 상관관계이므로 인과 확정에는 DOE 검증이 필요합니다.')
