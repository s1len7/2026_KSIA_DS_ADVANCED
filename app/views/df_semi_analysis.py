"""df_semi_analysis.csv — 공정 단계 기여도 · 중간 품질지표 · 고위험 조건 (analysis/df_semi_analysis.ipynb 웹 버전)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import PoissonRegressor, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text

from common import (BLUE, GRAY, INK, MUTED, ORANGE, RED, SERIES, caveat, fmt_table, load_csv, page_header, show,
                    show_describe, style_fig)

SOURCE = 'df_semi_analysis.csv'
HIGH = 195  # df_analysis 의 Defect 기준 (Target ≥ 195)
DROP = ['Target', 'Flux840s', 'Temp_Oxid', 'Vapor', 'Wavelength']
STAGES = {
    '산화': ['type', 'Temp_OXid', 'ppm', 'Pressure', 'Oxid_time', 'thickness'],
    '포토': ['resist_target', 'N2_HMDS', 'pressure_HMDS', 'temp_HMDS', 'temp_HMDS_bake', 'time_HMDS_bake',
             'spin1', 'spin2', 'spin3', 'photoresist_bake', 'temp_softbake', 'time_softbake', 'Line_CD',
             'UV_type', 'Resolution', 'Energy_Exposure'],
    '식각': ['Thin F4', 'Thin F3', 'Thin F2', 'Thin F1', 'Temp_Etching', 'Source_Power', 'Selectivity'],
    '이온주입': ['Flux60s', 'Flux90s', 'Flux160s', 'Flux480s', 'input_Energy', 'Temp_implantation'],
    '열처리': ['Furance_Temp', 'RTA_Temp'],
}
STAGE_OF = {c: st_name for st_name, cols in STAGES.items() for c in cols}
STAGE_COLOR = dict(zip(STAGES, SERIES))  # 고정 순서 배정


def hgb(**kw):
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0, **kw)


KF = KFold(5, shuffle=True, random_state=0)


# ---------------------------------------------------------------- 계산 (캐시)
@st.cache_data(show_spinner=False)
def features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=DROP).copy()
    X['type'] = (X['type'] == 'wet').astype(int)
    X['UV_type'] = X['UV_type'].map({'G': 436, 'H': 405, 'I': 365})  # 노광 파장(nm)
    return X.astype(float), df['Target']


@st.cache_data(show_spinner=False)
def relation_to_df_analysis(df: pd.DataFrame) -> dict:
    ref = load_csv('df_analysis.csv', encoding='cp949')
    common = [c for c in df.columns if c in ref.columns]
    recon = {
        'Defect': (df['Target'] >= HIGH).astype(int).astype(str),
        '두께정상': pd.Series(np.where(df['thickness'] >= 700, '정상', '비정상')),
        'CD정상': pd.Series(np.where(df['Line_CD'].between(22, 55), '정상', '비정상')),
    }
    return {
        'only_ref': [c for c in ref.columns if c not in df.columns],
        'identical': all((df[c].astype(str).values == ref[c].astype(str).values).all() for c in common),
        'n_common': len(common),
        'recon': {k: (v.values == ref[k].astype(str).values).mean() for k, v in recon.items()},
    }


@st.cache_data(show_spinner='공정 단계별 기여도 계산 중... (최초 1회)')
def stage_attribution(X: pd.DataFrame, y: pd.Series) -> tuple[float, pd.DataFrame]:
    full = cross_val_score(hgb(), X, y, cv=KF, scoring='r2', n_jobs=-1).mean()
    rows = []
    for name, cols in STAGES.items():
        alone = cross_val_score(hgb(), X[cols], y, cv=KF, scoring='r2', n_jobs=-1).mean()
        without = cross_val_score(hgb(), X.drop(columns=cols), y, cv=KF, scoring='r2', n_jobs=-1).mean()
        rows.append((name, len(cols), alone, without - full))
    return full, pd.DataFrame(rows, columns=['단계', '변수 수', '단독 R²', '제거 시 ΔR²']).set_index('단계')


@st.cache_data(show_spinner='모델 비교 중...')
def model_compare(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    models = {
        'Ridge': make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))),
        'Poisson GLM': make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=2000)),
        'HistGB (MSE)': hgb(),
        'HistGB (Poisson)': hgb(loss='poisson'),
    }
    rows = []
    for name, m in models.items():
        p = cross_val_predict(m, X, y, cv=KF, n_jobs=-1)
        rows.append((name, r2_score(y, p), mean_absolute_error(y, p)))
    return pd.DataFrame(rows, columns=['모델', 'CV R²', 'CV MAE']).set_index('모델')


@st.cache_data(show_spinner='변수 중요도 계산 중...')
def importance(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    m = hgb().fit(Xtr, ytr)
    pi = permutation_importance(m, Xte, yte, n_repeats=15, random_state=0, scoring='r2', n_jobs=-1)
    return pd.Series(pi.importances_mean, X.columns).sort_values(ascending=False)


@st.cache_data(show_spinner='중간 품질지표 모델 계산 중...')
def intermediate_r2(X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    ox_in = ['type', 'Temp_OXid', 'ppm', 'Pressure', 'Oxid_time']
    ph_in = [c for c in STAGES['포토'] if c != 'Line_CD']
    cases = [('thickness', '산화 입력 5개', ox_in),
             ('thickness', '전체 변수', [c for c in X.columns if c != 'thickness']),
             ('Line_CD', '포토 입력 15개', ph_in),
             ('Line_CD', '전체 변수', [c for c in X.columns if c != 'Line_CD'])]
    rows = [(t, lab, cross_val_score(hgb(), X[cols], df[t], cv=KF, scoring='r2', n_jobs=-1).mean()) for t, lab, cols in cases]
    return pd.DataFrame(rows, columns=['지표', '입력', 'CV R²'])


@st.cache_data(show_spinner=False)
def tree_rules(X: pd.DataFrame, y: pd.Series, depth: int) -> tuple[float, str, pd.DataFrame]:
    params = dict(max_depth=depth, min_samples_leaf=20, random_state=0)
    r2 = cross_val_score(DecisionTreeRegressor(**params), X, y, cv=KF, scoring='r2', n_jobs=-1).mean()
    tree = DecisionTreeRegressor(**params).fit(X, y)
    leaf = pd.Series(tree.apply(X), index=y.index)
    tbl = y.groupby(leaf).agg(n='size', 평균_Target='mean', 비율_195이상=lambda v: (v >= HIGH).mean())
    tbl.index.name = 'leaf'
    return r2, export_text(tree, feature_names=list(X.columns), decimals=4), tbl.sort_values('평균_Target', ascending=False)


def quantile_table(df: pd.DataFrame, col: str, q: int) -> pd.DataFrame:
    bins = pd.qcut(df[col], q, duplicates='drop')
    g = df.groupby(bins, observed=True)['Target'].agg(평균='mean', 중앙값='median', n='size',
                                                       비율=lambda v: (v >= HIGH).mean())
    g.index = [f'{iv.left:,.4f} ~ {iv.right:,.4f}' for iv in g.index]
    return g


# ---------------------------------------------------------------- 페이지
page_header('df_semi_analysis — 공정 단계 기여도',
            'df_analysis 에서 파생 라벨을 뺀 원본형 데이터로, Target 을 공정 단계·중간 품질지표 관점에서 분석',
            SOURCE, 'df_semi_analysis.ipynb')

df = load_csv(SOURCE)
X, y = features(df)
full_r2, stage_tbl = stage_attribution(X, y)
etch_drop = stage_tbl.loc['식각', '제거 시 ΔR²']

k = st.columns(5)
k[0].metric('웨이퍼 수', f'{len(df):,}', f'{df.shape[1]}개 컬럼', delta_color='off')
k[1].metric('Target 평균', f'{y.mean():,.4f}', f'중앙값 {y.median():,.4f}', delta_color='off')
k[2].metric(f'Target ≥ {HIGH} 비율', f'{(y >= HIGH).mean():.1%}', f'{(y >= HIGH).sum()}개 웨이퍼', delta_color='off')
k[3].metric('HistGB CV R²', f'{full_r2:,.4f}', '전체 37개 변수', delta_color='off')
k[4].metric('식각 단계 제거 시', f'{etch_drop:+,.4f}', 'ΔR² (유일한 핵심 단계)', delta_color='off')

tabs = st.tabs(['데이터 개요', 'Target 분포', '공정 단계 기여도', '중간 품질지표', '영향 인자·규칙', '고위험 조건'])

# ---- 1. 데이터 개요
with tabs[0]:
    rel = relation_to_df_analysis(df)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader('df_analysis.csv 와의 관계')
        st.markdown(
            f'- 공통 **{rel["n_common"]}개** 컬럼의 행 순서·값 완전 일치: **{rel["identical"]}**\n'
            f'- df_analysis 에만 있는 컬럼: `{"`, `".join(rel["only_ref"])}` (파생 라벨)\n'
            f'- 결측치 **{df.isna().sum().sum()}개**, 중복 행 **{df.duplicated().sum()}개**')
        recon = pd.DataFrame({
            '재현 규칙': [f'Target ≥ {HIGH}', 'thickness ≥ 700', '22 ≤ Line_CD ≤ 55'],
            '일치율': [f'{v:.2%}' for v in rel['recon'].values()]}, index=list(rel['recon'].keys()))
        st.dataframe(recon, width='stretch')
    with c2:
        st.subheader('범주형 변수 분포')
        for c in ['type', 'UV_type']:
            vc = df[c].value_counts()
            fig = go.Figure(go.Bar(x=vc.values, y=vc.index, orientation='h', marker_color=BLUE,
                                   text=[f'{v:,} ({v / len(df):.1%})' for v in vc.values], textposition='outside',
                                   hovertemplate='%{y}: %{x:,}개<extra></extra>'))
            fig.update_xaxes(range=[0, vc.max() * 1.3])
            show(style_fig(fig, height=170, legend=False).update_layout(title=c))
        st.caption('`Vapor` 는 `type` 과, `Wavelength` 는 `UV_type` 과 1:1 대응 · `Flux840s`, `Temp_Oxid` 는 상수 → 모델 입력에서 제외')
    st.subheader('기술통계')
    show_describe(df)

# ---- 2. Target 분포
with tabs[1]:
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        fig.add_histogram(x=y[y < HIGH], xbins=dict(start=0, end=500, size=10), marker_color=BLUE, name=f'Target < {HIGH}',
                          marker_line=dict(color='white', width=1))
        fig.add_histogram(x=y[y >= HIGH], xbins=dict(start=0, end=500, size=10), marker_color=RED, name=f'Target ≥ {HIGH}',
                          marker_line=dict(color='white', width=1))
        fig.add_vline(x=HIGH - 0.5, line_dash='dash', line_color=INK, annotation_text=f'{HIGH}', annotation_position='top right')
        fig.update_layout(barmode='overlay', title='Target 분포 (10 단위 구간)')
        fig.update_xaxes(title='Target'); fig.update_yaxes(title='웨이퍼 수')
        show(style_fig(fig, height=400))
    with c2:
        qs = y.quantile([0.5, 0.9, 0.95, 0.99])
        st.dataframe(pd.DataFrame({
            '값': [f'{y.mean():,.4f}', f'{y.var():,.4f}', f'{y.skew():,.4f}', f'{(y == 0).sum()}']
                  + [f'{v:,.4f}' for v in qs.values]},
            index=['평균', '분산', '왜도', '0 값 개수'] + [f'{q:.0%} 분위수' for q in qs.index]), width='stretch')
        caveat(f'Target 은 정수형이고 분산({y.var():,.4f})이 평균({y.mean():,.4f})보다 훨씬 커 **과산포 카운트형** 지표로 보입니다. '
               '오른쪽 꼬리가 길어 평균보다 분위수·비율로 보는 것이 안정적입니다.')
    ct = pd.crosstab(df['type'], df['UV_type'], df['Target'], aggfunc='mean')
    p_uv = stats.kruskal(*[g['Target'] for _, g in df.groupby('UV_type')]).pvalue
    p_type = stats.mannwhitneyu(*[g['Target'] for _, g in df.groupby('type')]).pvalue
    fig = go.Figure(go.Heatmap(z=ct.values, x=ct.columns, y=ct.index, colorscale='Blues', texttemplate='%{z:,.4f}',
                               hovertemplate='type %{y} · UV %{x}<br>평균 Target %{z:,.4f}<extra></extra>'))
    fig.update_layout(title=f'type × UV_type 평균 Target  (UV Kruskal p={p_uv:,.4f}, type Mann-Whitney p={p_type:,.4f})')
    show(style_fig(fig, height=260, legend=False))

# ---- 3. 공정 단계 기여도
with tabs[2]:
    st.markdown('- **단독 R²**: 해당 단계 변수만으로 Target 을 예측한 5-fold CV R²\n'
                '- **제거 시 ΔR²**: 전체 모델에서 그 단계를 뺐을 때 R² 변화 — 다른 단계로 **대체 불가능한 고유 정보**')
    order = stage_tbl.sort_values('단독 R²').index
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(x=stage_tbl.loc[order, '단독 R²'], y=order, orientation='h',
                               marker_color=[STAGE_COLOR[s] for s in order],
                               text=[f'{v:,.4f}' for v in stage_tbl.loc[order, '단독 R²']], textposition='outside',
                               hovertemplate='%{y}: R² %{x:,.4f}<extra></extra>'))
        fig.update_xaxes(range=[0, 1], title='CV R²')
        show(style_fig(fig, height=330, legend=False).update_layout(title='단계 변수만으로 예측한 R²'))
    with c2:
        fig = go.Figure(go.Bar(x=stage_tbl.loc[order, '제거 시 ΔR²'], y=order, orientation='h',
                               marker_color=[STAGE_COLOR[s] for s in order],
                               text=[f'{v:+,.4f}' for v in stage_tbl.loc[order, '제거 시 ΔR²']], textposition='outside',
                               hovertemplate='%{y}: ΔR² %{x:+,.4f}<extra></extra>'))
        fig.update_xaxes(range=[-0.18, 0.03], title=f'ΔR²  (전체 R² = {full_r2:,.4f})')
        show(style_fig(fig, height=330, legend=False).update_layout(title='해당 단계 제거 시 R² 변화'))
    st.dataframe(stage_tbl.assign(변수=[', '.join(STAGES[s]) for s in stage_tbl.index]).style.format(
        {'단독 R²': '{:,.4f}', '제거 시 ΔR²': '{:+,.4f}'}), width='stretch')
    caveat('식각 단계만 제거 시 R² 가 크게 떨어집니다. 이온주입·열처리는 단독 R² 가 있어도 제거 영향이 없어, '
           '다른 단계와 **겹치는 정보**만 가진 것으로 보입니다. 단계 간 상관이 있어 "단독 R²" 만으로 인과를 판단하면 안 됩니다.')

# ---- 4. 중간 품질지표
with tabs[3]:
    c1, c2 = st.columns([2, 3])
    with c1:
        st.subheader('중간 지표 예측력')
        st.dataframe(fmt_table(intermediate_r2(X, df), {'CV R²': '{:,.4f}'}), width='stretch', hide_index=True)
        sp = df.select_dtypes('number').drop(columns=['Flux840s', 'Temp_Oxid']).corr(method='spearman')
        tgt = st.radio('Spearman 상관 상위 변수', ['thickness', 'Line_CD'], horizontal=True)
        top = sp[tgt].drop([tgt, 'Target']).sort_values(key=abs, ascending=False).head(6)
        st.dataframe(pd.DataFrame({'단계': [STAGE_OF.get(c, '') for c in top.index], 'Spearman ρ': top.values},
                                  index=top.index).style.format({'Spearman ρ': '{:+,.4f}'}), width='stretch')
    with c2:
        st.subheader('변수 구간별 Target')
        num_cols = [c for c in X.columns if c not in ('type', 'UV_type') and df[c].nunique() > 10]
        cc = st.columns([3, 2])
        var = cc[0].selectbox('변수 선택', num_cols, index=num_cols.index('thickness'))
        q = cc[1].slider('구간 수 (분위)', 3, 10, 5)
        g = quantile_table(df, var, q)
        fig = go.Figure(go.Bar(x=g.index, y=g['평균'], marker_color=[RED if v == g['평균'].max() else BLUE for v in g['평균']],
                               customdata=np.c_[g['n'], g['비율'] * 100, g['중앙값']],
                               text=[f'{v:,.4f}' for v in g['평균']], textposition='outside',
                               hovertemplate='%{x}<br>평균 %{y:,.4f} · 중앙값 %{customdata[2]:,.4f}<br>'
                                             'n=%{customdata[0]} · ≥195 %{customdata[1]:.1f}%<extra></extra>'))
        fig.add_hline(y=y.mean(), line_dash='dot', line_color=INK, annotation_text=f'전체 평균 {y.mean():,.4f}',
                      annotation_position='top left')
        fig.update_yaxes(title='평균 Target', range=[0, g['평균'].max() * 1.2])
        fig.update_xaxes(title=f'{var} 구간 ({STAGE_OF.get(var, "")})')
        show(style_fig(fig, height=360, legend=False).update_layout(title=f'{var} {q}분위 구간별 평균 Target'))
    caveat('Line_CD 는 포토 입력만으로 잘 예측되지만, thickness 는 산화 입력만으로는 설명력이 낮고 식각 온도 등 '
           '후공정 변수와 상관이 더 큽니다. 측정 시점·순서 정보가 없어 후공정 변수와의 상관은 해석에 주의가 필요합니다.')

# ---- 5. 영향 인자·규칙
with tabs[4]:
    c1, c2 = st.columns(2)
    with c1:
        imp = importance(X, y).head(12)[::-1]
        fig = go.Figure()
        for s_name in STAGES:  # 단계별 trace → 범례 = 단계
            m = [STAGE_OF[c] == s_name for c in imp.index]
            if any(m):
                fig.add_bar(x=imp.values[m], y=imp.index[m], orientation='h', name=s_name, marker_color=STAGE_COLOR[s_name],
                            hovertemplate='%{y}: ΔR² %{x:,.4f}<extra>' + s_name + '</extra>')
        fig.update_yaxes(categoryorder='array', categoryarray=list(imp.index))
        fig.update_xaxes(title='Permutation importance (R² 감소량)')
        show(style_fig(fig, height=420).update_layout(title='Target 변수 중요도 상위 12', barmode='overlay'))
    with c2:
        f2 = pd.qcut(df['Thin F2'], 3, labels=['F2 하', 'F2 중', 'F2 상'])
        f4 = pd.qcut(df['Thin F4'], 3, labels=['F4 하', 'F4 중', 'F4 상'])
        heat = pd.crosstab(f2, f4, df['Target'], aggfunc='mean')
        cnt = pd.crosstab(f2, f4)
        fig = go.Figure(go.Heatmap(z=heat.values, x=list(heat.columns), y=list(heat.index), colorscale='Blues',
                                   texttemplate='%{z:,.4f}', customdata=cnt.values,
                                   hovertemplate='%{y} · %{x}<br>평균 Target %{z:,.4f}<br>n=%{customdata}<extra></extra>'))
        fig.update_yaxes(autorange='reversed')
        show(style_fig(fig, height=420, legend=False).update_layout(title='Thin F2 × Thin F4 3분위 조합별 평균 Target'))
    st.subheader('모델 비교 (5-fold CV)')
    st.dataframe(fmt_table(model_compare(X, y), {'CV R²': '{:,.4f}', 'CV MAE': '{:,.4f}'}), width='stretch')
    st.subheader('Regression Tree 규칙')
    depth = st.slider('트리 깊이 (max_depth)', 2, 5, 3)
    r2_t, rules, leaf_tbl = tree_rules(X, y, depth)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.caption(f'max_depth={depth} · CV R² = {r2_t:,.4f} · value = 해당 잎의 평균 Target')
        st.code(rules, language=None)
    with c2:
        st.dataframe(fmt_table(leaf_tbl, {'평균_Target': '{:,.4f}', '비율_195이상': '{:.1%}'}), width='stretch')
    caveat('트리 모델(HistGB)이 선형·Poisson 모델보다 R² 가 크게 높아 **임계값형 비선형 구조**가 있습니다. '
           '단일 결정트리는 해석용이며 예측 성능은 HistGB 보다 낮습니다.')

# ---- 6. 고위험 조건
with tabs[5]:
    presets = {
        '산화온도<900 & F4>458': (df['Temp_OXid'] < 900) & (df['Thin F4'] > 458),
        'F2>3666.5 & ppm≤22.45': (df['Thin F2'] > 3666.5) & (df['ppm'] <= 22.45),
        'F4>458 & ppm≤28.45': (df['Thin F4'] > 458) & (df['ppm'] <= 28.45),
        'F2>3656.5 & F4>458': (df['Thin F2'] > 3656.5) & (df['Thin F4'] > 458),
        'F2>3666.5': df['Thin F2'] > 3666.5,
        '전체': pd.Series(True, index=df.index),
    }
    risk = pd.DataFrame({'n': {k_: m.sum() for k_, m in presets.items()},
                         '평균 Target': {k_: y[m].mean() for k_, m in presets.items()},
                         f'≥{HIGH} 비율': {k_: (y[m] >= HIGH).mean() for k_, m in presets.items()}})
    rk = risk.sort_values('평균 Target')
    fig = go.Figure(go.Bar(x=rk['평균 Target'], y=rk.index, orientation='h',
                           marker_color=[RED if v >= 250 else (BLUE if v >= 130 else GRAY) for v in rk['평균 Target']],
                           text=[f'{v:,.4f} (n={n:,}, ≥{HIGH} {p:.0%})' for v, n, p in zip(rk['평균 Target'], rk['n'], rk[f'≥{HIGH} 비율'])],
                           textposition='outside', hovertemplate='%{y}<br>평균 Target %{x:,.4f}<extra></extra>'))
    fig.add_vline(x=HIGH, line_dash='dot', line_color=INK)
    fig.update_xaxes(range=[0, 560], title=f'평균 Target (점선 = {HIGH})')
    show(style_fig(fig, height=340, legend=False).update_layout(title='대표 고위험 조건 (조건 간 중복 있음)'))

    st.subheader('조건 직접 설정')
    st.caption('슬라이더로 조건을 바꾸면 해당 웨이퍼 수와 Target 수준이 즉시 계산됩니다.')

    def rng(c):
        return float(df[c].min()), float(df[c].max())

    s = st.columns(4)
    f2_min = s[0].slider('Thin F2 초과', *rng('Thin F2'), 3666.5, step=0.5)
    f4_min = s[1].slider('Thin F4 초과', *rng('Thin F4'), float(df['Thin F4'].min()), step=1.0)
    ppm_max = s[2].slider('ppm 이하', *rng('ppm'), 22.45, step=0.05)
    tox_max = s[3].slider('Temp_OXid 이하', *rng('Temp_OXid'), float(df['Temp_OXid'].max()), step=1.0)
    mask = (df['Thin F2'] > f2_min) & (df['Thin F4'] > f4_min) & (df['ppm'] <= ppm_max) & (df['Temp_OXid'] <= tox_max)

    n_sel = int(mask.sum())
    m = st.columns(4)
    m[0].metric('해당 웨이퍼 수', f'{n_sel:,}', f'전체의 {n_sel / len(df):.1%}', delta_color='off')
    if n_sel:
        m[1].metric('평균 Target', f'{y[mask].mean():,.4f}', f'{y[mask].mean() - y.mean():+,.4f} vs 전체', delta_color='inverse')
        m[2].metric(f'Target ≥ {HIGH} 비율', f'{(y[mask] >= HIGH).mean():.1%}',
                    f'{((y[mask] >= HIGH).mean() - (y >= HIGH).mean()) * 100:+.1f}%p vs 전체', delta_color='inverse')
        m[3].metric('Target 중앙값', f'{y[mask].median():,.4f}', f'전체 {y.median():,.4f}', delta_color='off')
    else:
        m[1].metric('평균 Target', '—')
        st.warning('조건에 해당하는 웨이퍼가 없습니다. 슬라이더 범위를 넓혀 보세요.')
    if 0 < n_sel < 30:
        st.warning(f'표본이 {n_sel}개로 적어 비율이 불안정합니다.')

    sel, rest = df[mask], df[~mask]
    fig = go.Figure()
    fig.add_scatter(x=rest['Thin F2'], y=rest['Thin F4'], mode='markers', name='조건 밖',
                    marker=dict(color=GRAY, size=6, opacity=0.45), hovertemplate='F2 %{x} · F4 %{y}<extra>조건 밖</extra>')
    for lab, part, col in [(f'조건 안 · Target < {HIGH}', sel[sel['Target'] < HIGH], BLUE),
                           (f'조건 안 · Target ≥ {HIGH}', sel[sel['Target'] >= HIGH], RED)]:
        fig.add_scatter(x=part['Thin F2'], y=part['Thin F4'], mode='markers', name=lab, customdata=part['Target'],
                        marker=dict(color=col, size=9, line=dict(color='white', width=1.5)),
                        hovertemplate='F2 %{x} · F4 %{y}<br>Target %{customdata}<extra></extra>')
    fig.update_xaxes(title='Thin F2'); fig.update_yaxes(title='Thin F4')
    show(style_fig(fig, height=420).update_layout(title='Thin F2 × Thin F4 산점도 (조건 강조)'))
    caveat('관측 데이터 기반 상관관계이므로, 조건이 Target 을 높인다는 **인과 확정에는 DOE 검증**이 필요합니다. '
           '대표 고위험 조건은 n=15~25 로 작아 신뢰구간이 넓습니다.')
