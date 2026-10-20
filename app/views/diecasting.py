"""DieCasting_analysis.csv — 다이캐스팅 공정 불량 분석 (analysis/DieCasting_analysis.ipynb 대응 페이지)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from common import (BLUE, GRAY, INK, MUTED, ORANGE, RED, caveat, fmt_table, load_csv, page_header, show,
                    show_describe, style_fig)

SOURCE = 'DieCasting_analysis.csv'
SENSOR = ['High_Velocity', 'Rapid_Rise_Time', 'Biscuit_Thickness', 'Pressure_Rise_Time']
PERIODS = ['Q1', 'Q2', 'Q3', 'Q4']


# ───────────────────────── 데이터 준비 ─────────────────────────
@st.cache_data
def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """센서 0값 행 제거, 상수 컬럼 제거, 쇼트 순서·제품별 기간(Q1~Q4) 추가."""
    zero = (raw[SENSOR] == 0).any(axis=1)
    df = raw.loc[~zero].drop(columns=['Velocity_1']).reset_index(drop=True)
    df['shot'] = np.arange(len(df))
    df['period'] = ''
    for p in (1, 2):
        m = df.Product_Type == p
        df.loc[m, 'period'] = pd.qcut(df.loc[m, 'shot'], 4, labels=PERIODS).astype(str)
    return df


@st.cache_data
def quality_summary(raw: pd.DataFrame) -> dict:
    feats = [c for c in raw.columns if c != 'Defect']
    grp = raw.groupby(feats)['Defect'].agg(['nunique', 'size'])
    conflict = grp[grp['nunique'] > 1]
    zero = (raw[SENSOR] == 0).any(axis=1)
    return {
        'const': [c for c in raw.columns if raw[c].nunique() == 1],
        'zero_rows': raw.loc[zero, SENSOR + ['Cylinder_Pressure', 'Casting_Pressure', 'Defect']],
        'dup_rows': int(raw.duplicated().sum()),
        'same_prev': int(raw[feats].eq(raw[feats].shift()).all(axis=1).sum()),
        'unique_x': len(grp),
        'conflict_groups': len(conflict),
        'conflict_rows': int(conflict['size'].sum()),
        'autocorr': float(raw.Defect.autocorr(1)),
    }


def num_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ('Defect', 'Product_Type', 'shot', 'period')]


@st.cache_data
def mann_whitney(df: pd.DataFrame, product: int) -> pd.DataFrame:
    s = df[df.Product_Type == product]
    rows = []
    for col in num_cols(df):
        a, b = s.loc[s.Defect == 0, col], s.loc[s.Defect == 1, col]
        rows.append((col, a.mean(), b.mean(), s[col].corr(s.Defect, method='spearman'),
                     stats.mannwhitneyu(a, b).pvalue))
    return (pd.DataFrame(rows, columns=['변수', '평균_정상', '평균_불량', 'Spearman', 'p_value'])
            .sort_values('p_value').reset_index(drop=True))


@st.cache_data(show_spinner='모델 교차검증 중... (처음 한 번만 실행)')
def cv_results(df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = df.drop(columns=['Defect', 'shot', 'period']).astype(float)
    y = df['Defect']
    block = df['shot'] * 10 // len(df)
    models = {
        'Logistic': make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=3000)),
        'RandomForest': RandomForestClassifier(200, min_samples_leaf=5, class_weight='balanced', n_jobs=-1, random_state=0),
        'HistGB': HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, class_weight='balanced', random_state=0),
    }
    cvs = {'랜덤 5-fold': (StratifiedKFold(5, shuffle=True, random_state=0), None),
           '시간 블록 5-fold': (GroupKFold(5), block)}
    rows = []
    for mname, m in models.items():
        for cname, (cv, groups) in cvs.items():
            p = cross_val_predict(m, X, y, cv=cv, groups=groups, method='predict_proba')[:, 1]
            rows.append((mname, cname, roc_auc_score(y, p), average_precision_score(y, p)))
    return pd.DataFrame(rows, columns=['모델', 'CV', 'ROC-AUC', 'PR-AUC'])


@st.cache_data(show_spinner='시간 순 hold-out 평가 중...')
def time_holdout(df: pd.DataFrame) -> tuple[float, int, pd.Series]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score

    X = df.drop(columns=['Defect', 'shot', 'period']).astype(float)
    y = df['Defect']
    cut = int(len(df) * 0.75)
    rf = RandomForestClassifier(200, min_samples_leaf=5, class_weight='balanced', n_jobs=-1, random_state=0)
    rf.fit(X.iloc[:cut], y.iloc[:cut])
    auc = roc_auc_score(y.iloc[cut:], rf.predict_proba(X.iloc[cut:])[:, 1])
    pi = permutation_importance(rf, X.iloc[cut:], y.iloc[cut:], n_repeats=5, random_state=0, scoring='roc_auc')
    return auc, len(df) - cut, pd.Series(pi.importances_mean, X.columns).sort_values(ascending=False)


@st.cache_data
def tree_text(df: pd.DataFrame, product: int) -> str:
    from sklearn.tree import DecisionTreeClassifier, export_text
    s = df[df.Product_Type == product]
    proc = [c for c in num_cols(df) if not c.startswith('Factory')]
    t = DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, random_state=0).fit(s[proc], s.Defect)
    return export_text(t, feature_names=proc, show_weights=True, decimals=4)


def rate_n(s: pd.DataFrame) -> tuple[float, int]:
    return (s.Defect.mean() if len(s) else np.nan), len(s)


def pct(v: float) -> str:
    return '–' if pd.isna(v) else f'{v:.1%}'


# ───────────────────────── 페이지 ─────────────────────────
page_header('DieCasting — 다이캐스팅 공정 불량 분석',
            '쇼트(shot) 단위 공정 조건과 불량 여부 · 제품별 비교, 시간 드리프트, 누수 없는 모델 평가',
            SOURCE, 'DieCasting_analysis.ipynb')

raw = load_csv(SOURCE)
df = prepare(raw)
q = quality_summary(raw)
P = {p: df[df.Product_Type == p] for p in (1, 2)}

k = st.columns(5)
k[0].metric('쇼트 수 (정제 후)', f'{len(df):,}', f'원본 {len(raw):,}', delta_color='off')
k[1].metric('전체 불량률', f'{df.Defect.mean():.1%}')
k[2].metric('Product 1 불량률', f'{P[1].Defect.mean():.1%}', f'n={len(P[1]):,}', delta_color='off')
k[3].metric('Product 2 불량률', f'{P[2].Defect.mean():.1%}', f'n={len(P[2]):,}', delta_color='off')
k[4].metric('직전 쇼트와 동일한 행', f'{q["same_prev"] / len(raw):.1%}', f'{q["same_prev"]:,}행', delta_color='off')

tabs = st.tabs(['데이터 개요', '제품별 불량률', '기간(시간) 효과', '모델: 랜덤 CV vs 시간 블록 CV', '영향 인자', '규칙·고위험 조건'])

# ── 1. 데이터 개요 ──
with tabs[0]:
    st.subheader('기술통계 (원본)')
    show_describe(raw)

    st.subheader('데이터 품질 점검')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('완전 중복 행', f'{q["dup_rows"]:,}')
    c2.metric('고유 공정값 조합', f'{q["unique_x"]:,}')
    c3.metric('같은 공정값·다른 라벨', f'{q["conflict_groups"]}조합', f'{q["conflict_rows"]}행', delta_color='off')
    c4.metric('불량 자기상관 (lag 1)', f'{q["autocorr"]:,.4f}')
    st.markdown(
        f'- **상수 컬럼**: `{", ".join(q["const"])}` → 제거\n'
        f'- **센서 0값 행 {len(q["zero_rows"])}개** (`{"`, `".join(SENSOR)}` 중 하나가 0) → 측정 이상으로 제거\n'
        f'- 공정값이 여러 쇼트 동안 그대로 기록되고, 불량도 연속해서 발생 → 무작위로 섞는 검증은 성능을 부풀림'
    )
    with st.expander(f'센서 0값 행 {len(q["zero_rows"])}개 보기'):
        st.dataframe(fmt_table(q['zero_rows']), width='stretch')
    caveat('행 순서를 **쇼트 생산 순서**로 가정했습니다. 데이터에 시간 컬럼이 없으므로 '
           '기간·시간 블록 분석은 이 가정에 의존합니다.')

# ── 2. 제품별 불량률 ──
with tabs[1]:
    by = df.groupby('Product_Type')['Defect'].agg(['mean', 'size'])
    pval = stats.chi2_contingency(pd.crosstab(df.Product_Type, df.Defect))[1]
    left, right = st.columns([1, 2])
    with left:
        fig = go.Figure(go.Bar(
            x=[f'Product {i}' for i in by.index], y=by['mean'] * 100, marker_color=[BLUE, ORANGE],
            text=[f'{r:.1%}<br>n={n:,}' for r, n in zip(by['mean'], by['size'])], textposition='outside',
            hovertemplate='%{x}<br>불량률 %{y:.1f}%<extra></extra>'))
        fig.update_yaxes(title='불량률 (%)', range=[0, by['mean'].max() * 140])
        show(style_fig(fig, legend=False).update_layout(title='제품별 불량률'))
        st.caption(f'카이제곱 검정 p-value = {pval:,.4f}')
    with right:
        var = st.selectbox('공정 변수', num_cols(df), index=num_cols(df).index('Casting_Pressure'), key='dc_prod_var')
        fig = go.Figure()
        for p, color in [(1, BLUE), (2, ORANGE)]:
            fig.add_trace(go.Box(y=P[p][var], name=f'Product {p}', marker_color=color, boxmean=True))
        show(style_fig(fig, legend=False).update_layout(title=f'{var} 제품별 분포'))
    st.markdown('**제품별 공정값 중앙값** — 두 제품은 주조압력·형체력·용탕온도 등 공정 조건 자체가 다르므로 이후 분석은 제품별로 봅니다.')
    med = df.groupby('Product_Type')[num_cols(df)].median().T
    med.columns = [f'Product {c}' for c in med.columns]
    st.dataframe(med.style.format('{:,.4f}'), width='stretch')

# ── 3. 기간(시간) 효과 ──
with tabs[2]:
    win = st.slider('이동평균 창 (쇼트 수)', 50, 500, 200, 50, key='dc_roll')
    roll = df.Defect.rolling(win).mean() * 100
    fig = go.Figure()
    fig.add_vrect(x0=0, x1=len(P[1]), fillcolor='#ecebe7', line_width=0, layer='below',
                  annotation_text='Product 1', annotation_position='top left')
    fig.add_annotation(x=len(P[1]) + len(P[2]) / 2, y=1.0, yref='paper', text='Product 2', showarrow=False, font_color=MUTED)
    fig.add_trace(go.Scatter(x=df.shot, y=roll, mode='lines', line=dict(color=BLUE, width=2), name='이동평균 불량률',
                             hovertemplate='쇼트 %{x:,}<br>불량률 %{y:.1f}%<extra></extra>'))
    fig.add_hline(y=df.Defect.mean() * 100, line_dash='dot', line_color=INK,
                  annotation_text=f'전체 평균 {df.Defect.mean():.1%}', annotation_position='bottom right')
    fig.update_xaxes(title='쇼트 순서'); fig.update_yaxes(title='불량률 (%)')
    show(style_fig(fig, height=360, legend=False).update_layout(title=f'불량률 시간 추이 ({win}쇼트 이동평균)'))

    c1, c2 = st.columns([1, 3])
    prod = c1.radio('제품', [1, 2], format_func=lambda p: f'Product {p}', horizontal=True, key='dc_period_prod')
    s = P[prod]
    default_vars = {1: ['Coolant_Pressure', 'Factory_Humidity', 'Biscuit_Thickness'],
                    2: ['High_Velocity', 'Coolant_Pressure', 'Coolant_Temp', 'Melting_Furnace_Temp']}[prod]
    vars_ = c2.multiselect('함께 볼 공정 변수', num_cols(df), default=default_vars, key=f'dc_period_vars_{prod}')
    tbl = s.groupby('period')[['Defect'] + vars_].mean().reindex(PERIODS)
    tbl.insert(1, 'n', s.groupby('period').size().reindex(PERIODS))
    fig = go.Figure(go.Bar(x=PERIODS, y=tbl.Defect * 100, marker_color=BLUE if prod == 1 else ORANGE,
                           text=[f'{v:.1%}' for v in tbl.Defect], textposition='outside'))
    fig.update_yaxes(title='불량률 (%)', range=[0, tbl.Defect.max() * 130]); fig.update_xaxes(title='생산 기간 (제품 구간 4등분)')
    l, r = st.columns([1, 1])
    with l:
        show(style_fig(fig, height=320, legend=False).update_layout(title=f'Product {prod} 기간별 불량률'))
    with r:
        st.markdown('**기간별 불량률 & 공정값 평균**')
        st.dataframe(tbl.rename(columns={'Defect': '불량률'}).style.format({'불량률': '{:.1%}', 'n': '{:,.0f}'} |
                     {v: '{:,.4f}' for v in vars_}), width='stretch')
    caveat('**기간 교란** — Product 1 불량률은 기간마다 크게 줄고(Q1→Q4), 같은 시기에 습도·냉각수 압력 등도 함께 바뀝니다. '
           '전체 데이터에서 보이는 변수-불량 상관 중 상당수는 공정 변수가 아니라 시기(로트·설비 상태) 효과일 수 있습니다.')

# ── 4. 모델 ──
with tabs[3]:
    st.markdown('**시간 블록 CV**: 전체를 연속된 10개 구간으로 나누고 구간 단위로 train/test 를 분리해 인접 쇼트 누수를 막습니다.')
    res = cv_results(df)
    l, r = st.columns([3, 2])
    with l:
        fig = go.Figure()
        for cname, color in [('랜덤 5-fold', GRAY), ('시간 블록 5-fold', BLUE)]:
            sub = res[res.CV == cname]
            fig.add_trace(go.Bar(x=sub['모델'], y=sub['ROC-AUC'], name=cname, marker_color=color,
                                 text=[f'{v:,.4f}' for v in sub['ROC-AUC']], textposition='outside'))
        fig.add_hline(y=0.5, line_dash='dot', line_color=INK, annotation_text='무작위 예측 0.5', annotation_position='bottom left')
        fig.update_yaxes(title='ROC-AUC', range=[0.4, 1.0]); fig.update_layout(barmode='group')
        show(style_fig(fig).update_layout(title='랜덤 CV vs 시간 블록 CV (ROC-AUC)'))
    with r:
        pv = res.pivot(index='모델', columns='CV', values=['ROC-AUC', 'PR-AUC'])
        pv.columns = [f'{m} · {c}' for m, c in pv.columns]
        st.dataframe(fmt_table(pv), width='stretch')
        st.caption(f'PR-AUC 기준선(불량률) = {df.Defect.mean():,.4f}')
        auc_ho, n_ho, imp = time_holdout(df)
        st.metric('시간 순 hold-out ROC-AUC (RandomForest)', f'{auc_ho:,.4f}',
                  f'앞 75% 학습 → 뒤 {n_ho:,} 쇼트 평가', delta_color='off')
    caveat('**랜덤 CV 과대평가** — 랜덤 CV 는 거의 같은 인접 쇼트가 train/test 에 섞여 트리 모델이 이를 외웁니다. '
           '시간 블록 CV 와 hold-out 성능이 실제 미래 구간 예측력에 가깝고, 공정 변수만으로는 예측력이 약합니다.')

# ── 5. 영향 인자 ──
with tabs[4]:
    c1, c2 = st.columns([1, 3])
    prod = c1.radio('제품', [1, 2], format_func=lambda p: f'Product {p}', horizontal=True, key='dc_drv_prod')
    mw = mann_whitney(df, prod)
    l, r = st.columns([1, 1])
    with l:
        st.markdown(f'**Product {prod} 정상 vs 불량 (Mann-Whitney U)**')
        st.dataframe(fmt_table(mw, {'평균_정상': '{:,.4f}', '평균_불량': '{:,.4f}', 'Spearman': '{:+,.4f}', 'p_value': '{:,.4f}'}),
                     width='stretch', height=420)
    with r:
        top = mw.head(10).iloc[::-1]
        fig = go.Figure(go.Bar(x=top.Spearman, y=top['변수'], orientation='h',
                               marker_color=[RED if v > 0 else BLUE for v in top.Spearman],
                               text=[f'{v:+,.4f}' for v in top.Spearman], textposition='outside'))
        fig.update_xaxes(title='Spearman 상관 (불량과)')
        show(style_fig(fig, height=420, legend=False).update_layout(title='p-value 상위 10개 변수의 상관 방향'))

    _, imp = time_holdout(df)[1:]
    st.markdown('**Permutation importance** (시간 순 hold-out, ΔROC-AUC) — 미래 구간에서 실제로 도움이 된 변수')
    top_imp = imp.head(8).iloc[::-1]
    fig = go.Figure(go.Bar(x=top_imp.values, y=top_imp.index, orientation='h',
                           marker_color=[BLUE if v > 0.005 else GRAY for v in top_imp.values],
                           text=[f'{v:,.4f}' for v in top_imp.values], textposition='outside'))
    fig.update_xaxes(title='ΔROC-AUC')
    show(style_fig(fig, height=320, legend=False))

    var = c2.selectbox('변수 분포 보기', mw['변수'].tolist(), key=f'dc_drv_var_{prod}')
    s = P[prod]
    fig = go.Figure()
    for d, color, name in [(0, GRAY, '정상'), (1, RED, '불량')]:
        fig.add_trace(go.Histogram(x=s.loc[s.Defect == d, var], name=name, marker_color=color, opacity=0.75,
                                   histnorm='percent', nbinsx=40))
    fig.update_layout(barmode='overlay'); fig.update_xaxes(title=var); fig.update_yaxes(title='비율 (%)')
    show(style_fig(fig, height=320).update_layout(title=f'Product {prod} · {var} 분포 (정상 vs 불량)'))

# ── 6. 규칙·고위험 조건 ──
with tabs[5]:
    st.markdown('**조건 탐색기** — 제품·변수·임계값을 정하면 조건 충족/미충족 쇼트의 불량률과, 같은 기간 안에서도 차이가 유지되는지 보여줍니다.')
    presets = {
        '직접 설정': None,
        'P2 · High_Velocity > 2.54 (기간 내에서도 유지)': (2, 'High_Velocity', '>', 2.54),
        'P1 · Biscuit_Thickness ≤ 11 (Q1~Q3 에서 일관)': (1, 'Biscuit_Thickness', '≤', 11.0),
        'P1 · Coolant_Pressure ≤ 2.66 (Q4 에 집중 — 교란)': (1, 'Coolant_Pressure', '≤', 2.66),
        'P1 · Factory_Humidity ≤ 54.85 (Q1 에 집중 — 교란)': (1, 'Factory_Humidity', '≤', 54.85),
        'P2 · Melting_Furnace_Temp ≤ 661.1': (2, 'Melting_Furnace_Temp', '≤', 661.1),
    }
    preset = st.selectbox('추천 규칙', list(presets), index=1, key='dc_preset')
    pre = presets[preset]
    c1, c2, c3, c4 = st.columns([1, 2, 1, 3])
    prod = c1.radio('제품', [1, 2], index=(pre[0] - 1) if pre else 0, format_func=lambda p: f'Product {p}',
                    key=f'dc_rule_prod_{preset}')
    cols = num_cols(df)
    var = c2.selectbox('변수', cols, index=cols.index(pre[1]) if pre else 0, key=f'dc_rule_var_{preset}')
    op = c3.radio('조건', ['≤', '>'], index=['≤', '>'].index(pre[2]) if pre else 0, key=f'dc_rule_op_{preset}')
    s = P[prod]
    vals = np.sort(s[var].unique())
    lo, hi = float(vals.min()), float(vals.max())
    default = float(pre[3]) if pre and pre[0] == prod and pre[1] == var else float(np.median(vals))
    default = min(max(default, lo), hi)
    step = float(np.min(np.diff(vals))) if len(vals) > 1 else 1.0
    thr = c4.slider('임계값', lo, hi, default, step=step, key=f'dc_rule_thr_{preset}_{prod}_{var}')

    mask = s[var] <= thr if op == '≤' else s[var] > thr
    r_in, n_in = rate_n(s[mask]); r_out, n_out = rate_n(s[~mask])
    m = st.columns(3)
    m[0].metric(f'조건 충족 ({var} {op} {thr:g})', pct(r_in), f'n={n_in:,}', delta_color='off')
    m[1].metric('조건 미충족', pct(r_out), f'n={n_out:,}', delta_color='off')
    m[2].metric('불량률 차이', '–' if pd.isna(r_in) or pd.isna(r_out) else f'{(r_in - r_out) * 100:+.1f}%p')

    period_sel = st.multiselect('비교할 기간', PERIODS, default=PERIODS, key='dc_rule_periods')
    rows = []
    for per in period_sel:
        sp = s[s.period == per]; mp = mask[s.period == per]
        ri, ni = rate_n(sp[mp]); ro, no = rate_n(sp[~mp])
        rows.append((per, ri, ni, ro, no))
    pt = pd.DataFrame(rows, columns=['기간', '충족_불량률', '충족_n', '미충족_불량률', '미충족_n'])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=pt['기간'], y=pt['미충족_불량률'] * 100, name='조건 미충족', marker_color=GRAY,
                         text=[f'{pct(r)}<br>n={n:,}' for r, n in zip(pt['미충족_불량률'], pt['미충족_n'])], textposition='outside'))
    fig.add_trace(go.Bar(x=pt['기간'], y=pt['충족_불량률'] * 100, name=f'{var} {op} {thr:g}', marker_color=RED,
                         text=[f'{pct(r)}<br>n={n:,}' for r, n in zip(pt['충족_불량률'], pt['충족_n'])], textposition='outside'))
    ymax = np.nanmax(pd.concat([pt['충족_불량률'], pt['미충족_불량률']]).to_numpy(dtype=float)) if len(pt) else 0.5
    fig.update_yaxes(title='불량률 (%)', range=[0, (ymax if np.isfinite(ymax) else 0.5) * 135 + 5])
    fig.update_xaxes(title='생산 기간'); fig.update_layout(barmode='group')
    show(style_fig(fig, height=380).update_layout(title=f'Product {prod} · 기간 내 비교'))
    st.caption('n 이 작은 기간(수십 개 이하)의 불량률은 변동이 큽니다. 조건이 특정 기간에만 몰려 있으면 기간 효과와 분리할 수 없습니다.')

    with st.expander('Decision Tree 규칙 (공정 변수만, max_depth=3, weights = [정상, 불량])'):
        t1, t2 = st.columns(2)
        t1.markdown('**Product 1**'); t1.code(tree_text(df, 1), language=None)
        t2.markdown('**Product 2**'); t2.code(tree_text(df, 2), language=None)
    caveat('관측 데이터 기반 상관 분석입니다. 기간(로트·설비 상태) 효과가 크므로, 인과 확정에는 기간을 통제한 실험(DOE)이 필요합니다.')
