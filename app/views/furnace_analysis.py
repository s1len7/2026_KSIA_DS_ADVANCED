"""Furnace_analysis.csv — 파생 이상치 라벨 검증과 이상 요인 분석."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (BLUE, GRAY, INK, ORANGE, RED, caveat, load_csv, page_header, show, show_describe,
                    style_fig)

NUM = ['Z1_Temp', 'Z1_OP1', 'ST',
       'Tube_Furnace1_OP', 'Tube_Furnace2_OP', 'Tube_Furnace3_OP', 'Tube_Furnace4_OP', 'Tube_Furnace_CP',
       'Tube_Furnace1_Temp', 'Tube_Furnace2_Temp', 'Tube_Furnace3_Temp', 'Tube_Furnace4_Temp']
WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


def short(name: str) -> str:
    return name.replace('Tube_Furnace', 'TF')


# ── 데이터 준비 ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner='Furnace_analysis 불러오는 중...')
def get_data() -> pd.DataFrame:
    df = load_csv('Furnace_analysis.csv').copy()
    df['dt'] = pd.to_datetime(df['Datetime'])
    df['run'] = (df['Code_Num'] != df['Code_Num'].shift()).cumsum()
    return df


@st.cache_data(show_spinner='원본과 비교 중...')
def relation_checks() -> dict:
    df = get_data()
    raw = load_csv('Furnace_Tube_Data.csv')
    rows = []
    for c in NUM:
        m = raw[c].isna()
        if not m.any():
            continue
        filled = df.loc[m, c].unique()
        rows.append({'컬럼': c, '결측(원본)': int(m.sum()), '대체값': filled[0], '원본 전체 평균': raw[c].mean(),
                     '평균으로 대체': bool(len(filled) == 1 and np.isclose(filled[0], raw[c].mean())),
                     '나머지 값 보존': bool(np.allclose(raw.loc[~m, c], df.loc[~m, c]))})
    q1, q3 = df[NUM].quantile(0.25), df[NUM].quantile(0.75)
    iqr = q3 - q1
    count = ((df[NUM] < q1 - 1.5 * iqr) | (df[NUM] > q3 + 1.5 * iqr)).sum(axis=1)
    return {
        'impute': pd.DataFrame(rows),
        'same_rows': len(raw) == len(df),
        'dt_same': bool((pd.to_datetime(raw['Datetime']) == df['dt']).all()),
        'dt_example': (raw['Datetime'].iloc[0], df['Datetime'].iloc[0]),
        'weekday_same': bool((df['dt'].dt.day_name() == df['요일']).all()),
        'count_match': int((count == df['이상치개수']).sum()),
        'label_match': bool(((df['이상치개수'] >= 3).astype(int) == df['정상여부']).all()),
    }


@st.cache_data
def outlier_flags(k: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """전역 IQR k배 경계로 센서별 이탈 여부와 경계값."""
    df = get_data()
    q1, q3 = df[NUM].quantile(0.25), df[NUM].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    flags = (df[NUM] < lo) | (df[NUM] > hi)
    fences = pd.DataFrame({'하한': lo, '상한': hi, '하한 이탈(%)': (df[NUM] < lo).mean() * 100,
                           '상한 이탈(%)': (df[NUM] > hi).mean() * 100})
    return flags, fences


@st.cache_data
def runwise_label(k: float, thr: int) -> pd.Series:
    df = get_data()

    def count(d):
        q1, q3 = d[NUM].quantile(0.25), d[NUM].quantile(0.75)
        r = q3 - q1
        return ((d[NUM] < q1 - k * r) | (d[NUM] > q3 + k * r)).sum(axis=1)

    return (df.groupby('run', group_keys=False).apply(count) >= thr).astype(int)


@st.cache_resource(show_spinner='Run 단위 GroupKFold 학습 중 (약 30초, 최초 1회)...')
def fit_model() -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict

    df = get_data()
    X, y, g = df[NUM], df['정상여부'], df['run']
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)
    proba = cross_val_predict(clf, X, y, cv=GroupKFold(n_splits=11), groups=g, method='predict_proba')[:, 1]
    per_run = []
    for r, d in pd.DataFrame({'y': y, 'p': proba, 'run': g}).groupby('run'):
        per_run.append({'run': r, 'Code_Num': df.loc[df['run'] == r, 'Code_Num'].iloc[0], '이상 행': int(d['y'].sum()),
                        'ROC-AUC': roc_auc_score(d['y'], d['p']) if 0 < d['y'].sum() < len(d) else np.nan})
    tr, te = g != 7, g == 7
    fitted = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0).fit(X[tr], y[tr])
    sample = X[te].sample(5000, random_state=0)
    pi = permutation_importance(fitted, sample, y[sample.index], scoring='average_precision', n_repeats=5,
                                random_state=0)
    return {'auc': roc_auc_score(y, proba), 'ap': average_precision_score(y, proba), 'base': y.mean(),
            'per_run': pd.DataFrame(per_run).set_index('run'),
            'importance': pd.Series(pi.importances_mean, NUM).sort_values()}


# ── 페이지 ───────────────────────────────────────────────────────────────────────
page_header('Furnace_analysis — 이상치 라벨',
            '원본 시계열에 결측 대체·파생 라벨(이상치개수, 정상여부, 요일)을 더한 데이터의 검증과 이상 요인',
            'Furnace_analysis.csv', 'Furnace_analysis.ipynb')

df = get_data()
abn = df['정상여부'] == 1

k = st.columns(5)
k[0].metric('행 수', f'{len(df):,}')
k[1].metric('이상 행 (정상여부=1)', f'{int(abn.sum()):,}', help='값 1 이 이상(비정상)입니다')
k[2].metric('이상 비율', f'{abn.mean():.2%}')
k[3].metric('Run', f"{df['run'].nunique()}개")
k[4].metric('최다 이상 Run', 'GroupG 15.80%', help='run 7 (2022-01-08) — CP 급락 구간')

caveat('**`정상여부` 는 이름과 반대로 1 = 이상, 0 = 정상** 입니다. '
       '센서 12개 중 전역 IQR 1.5배 경계를 벗어난 개수(`이상치개수`)가 3개 이상이면 1 로 정의된 규칙 기반 라벨입니다.')

tabs = st.tabs(['원본과의 관계', '이상 라벨', '이상 요인', 'Run별 이상률', '모델'])

# 1. 원본과의 관계 -------------------------------------------------------------------
with tabs[0]:
    chk = relation_checks()
    st.subheader('Furnace_Tube_Data → Furnace_analysis 변환 재현')
    c = st.columns(4)
    c[0].metric('행·순서 동일', '예' if chk['same_rows'] else '아니오')
    c[1].metric('Datetime 동일(파싱 후)', '예' if chk['dt_same'] else '아니오',
                help=f"'{chk['dt_example'][0]}' → '{chk['dt_example'][1]}' (초 :00 추가)")
    c[2].metric('이상치개수 재현', f"{chk['count_match']:,} / {len(df):,}")
    c[3].metric('정상여부 = (이상치개수 ≥ 3)', '일치' if chk['label_match'] else '불일치')

    st.markdown('**결측치 대체** — 7개 컬럼 343셀을 컬럼 전체 평균으로 채움')
    st.dataframe(chk['impute'].style.format({'대체값': '{:,.4f}', '원본 전체 평균': '{:,.4f}'}), width='stretch',
                 hide_index=True)
    st.markdown(f"- `요일` = Datetime 의 요일명: **{'일치' if chk['weekday_same'] else '불일치'}** (일요일 데이터 없음)")
    caveat('평균 대체는 시계열 흐름을 무시합니다. 특히 `Tube_Furnace1_OP` 결측 254개는 run 2·10 에 블록으로 몰려 있어 '
           '해당 구간이 75.31 로 평평하게 채워졌습니다.')

    with st.expander('기술통계 (센서 12개 + 파생 컬럼)'):
        show_describe(df[NUM + ['이상치개수', '정상여부']])

# 2. 이상 라벨 -----------------------------------------------------------------------
with tabs[1]:
    st.subheader('라벨 규칙 바꿔 보기')
    c1, c2 = st.columns(2)
    kk = c1.slider('IQR 배수 k (경계 = Q1 − k·IQR, Q3 + k·IQR)', 1.0, 3.0, 1.5, 0.25)
    thr = c2.slider('이상 판정 기준: 이탈 센서 수 ≥', 1, 6, 3)
    flags, fences = outlier_flags(kk)
    cnt = flags.sum(axis=1)
    new_label = (cnt >= thr)

    c = st.columns(3)
    c[0].metric('이상 비율 (선택 규칙)', f'{new_label.mean():.2%}', delta=f'{(new_label.mean() - abn.mean()) * 100:+.2f}%p',
                delta_color='off')
    c[1].metric('이상 행 (선택 규칙)', f'{int(new_label.sum()):,}')
    c[2].metric('원본 라벨과 일치율', f'{(new_label.astype(int) == df["정상여부"]).mean():.2%}')

    vc = cnt.value_counts().sort_index()
    fig = go.Figure(go.Bar(x=vc.index, y=vc.values, marker_color=[RED if i >= thr else GRAY for i in vc.index],
                           text=[f'{v:,}' for v in vc.values], textposition='outside'))
    fig.update_layout(xaxis_title='이탈 센서 수', yaxis_title='행 수 (log)', yaxis_type='log', xaxis_dtick=1)
    fig.add_vline(x=thr - 0.5, line=dict(color=INK, dash='dash', width=1))
    show(style_fig(fig, height=360, legend=False))
    st.caption(f'k = {kk:g}, 기준 ≥ {thr} · 원본 라벨은 k = 1.5, 기준 ≥ 3')

    st.markdown('**센서별 IQR 경계와 이탈 비율**')
    st.dataframe(fences.style.format({'하한': '{:,.3f}', '상한': '{:,.3f}', '하한 이탈(%)': '{:.2f}', '상한 이탈(%)': '{:.2f}'}),
                 width='stretch')

# 3. 이상 요인 -----------------------------------------------------------------------
with tabs[2]:
    flags15, _ = outlier_flags(1.5)
    drv = pd.DataFrame({'이상 행': flags15[abn].mean() * 100, '정상 행': flags15[~abn].mean() * 100})
    drv = drv.sort_values('이상 행')
    st.subheader('이상 행 vs 정상 행에서 센서별 IQR 이탈 비율')
    fig = go.Figure()
    fig.add_trace(go.Bar(y=[short(c) for c in drv.index], x=drv['정상 행'], orientation='h', name='정상 행', marker_color=GRAY))
    fig.add_trace(go.Bar(y=[short(c) for c in drv.index], x=drv['이상 행'], orientation='h', name='이상 행', marker_color=RED))
    fig.update_layout(barmode='group', xaxis_title='IQR 이탈 비율 (%)')
    show(style_fig(fig, height=460))
    st.markdown('- CP · Z1_OP1 · TF2_OP · TF3_OP · Z1_Temp 가 이상 행의 46~66% 에서 이탈\n'
                '- **튜브 온도 4개는 이탈 0건** → 온도는 이상 라벨에 전혀 기여하지 않음')

    st.subheader('GroupG Run(run 7): CP 급락과 이상 비율')
    g = df[df['run'] == 7].groupby(df['dt'].dt.floor('10min')).agg(CP=('Tube_Furnace_CP', 'mean'),
                                                                   rate=('정상여부', 'mean'))
    c1, c2 = st.columns(2)
    fig = go.Figure(go.Scatter(x=g.index, y=g['CP'], mode='lines+markers', line=dict(color=BLUE, width=2), marker_size=5))
    fig.add_hline(y=0.443, line=dict(color=INK, dash='dot', width=1), annotation_text='IQR 하한 0.443',
                  annotation_position='bottom left')
    fig.update_layout(title='CP 10분 평균', yaxis_range=[0, 0.55])
    with c1:
        show(style_fig(fig, height=320, legend=False))
    fig = go.Figure(go.Bar(x=g.index, y=g['rate'] * 100, marker_color=RED))
    fig.update_layout(title='이상 비율 (%) 10분 단위', yaxis_title='%')
    with c2:
        show(style_fig(fig, height=320, legend=False))

# 4. Run별 이상률 ---------------------------------------------------------------------
with tabs[3]:
    st.subheader('Run 별 이상 비율 — 전역 IQR vs Run별 IQR')
    c1, c2 = st.columns(2)
    rk = c1.slider('IQR 배수 k', 1.0, 3.0, 1.5, 0.25, key='run_k')
    rthr = c2.slider('이탈 센서 수 ≥', 1, 6, 3, key='run_thr')
    gflags, _ = outlier_flags(rk)
    glob = (gflags.sum(axis=1) >= rthr).astype(int)
    loc = runwise_label(rk, rthr)
    cmp = pd.DataFrame({'Code_Num': df.groupby('run')['Code_Num'].first(),
                        '전역 IQR': glob.groupby(df['run']).mean() * 100,
                        'Run별 IQR': loc.groupby(df['run']).mean() * 100})
    labels = [f'run {r}<br>{c}' for r, c in cmp['Code_Num'].items()]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=cmp['전역 IQR'], name='전역 IQR (정상여부 방식)', marker_color=BLUE))
    fig.add_trace(go.Bar(x=labels, y=cmp['Run별 IQR'], name='Run별 IQR', marker_color=ORANGE))
    fig.update_layout(barmode='group', yaxis_title='이상 비율 (%)')
    show(style_fig(fig, height=400))
    c = st.columns(3)
    c[0].metric('전체 이상 비율 · 전역', f'{glob.mean():.2%}')
    c[1].metric('전체 이상 비율 · Run별', f'{loc.mean():.2%}')
    c[2].metric('두 방식 일치율', f'{(glob == loc).mean():.2%}')
    caveat('전역 IQR 은 기간 전체의 경계 하나를 쓰므로 Run 마다 다른 **운전 조건(regime)** 도 이상으로 잡습니다. '
           'GroupG 는 전역 15.80% → Run별 0.00% (CP 하락이 Run 안에서는 "새 정상"), '
           'run 1·2 는 Tube_Furnace2_OP 가 59~60 으로 높은 운전 조건이라 전역 기준에서만 이상입니다.')

    st.subheader('범주별 이상 비율')
    c1, c2, c3 = st.columns(3)
    for col, box, order in [('Code_Num', c1, None), ('Process_Type', c2, None), ('요일', c3, WEEKDAYS)]:
        t = df.groupby(col)['정상여부'].agg(['mean', 'size'])
        t = t.reindex([o for o in order if o in t.index]) if order else t.sort_values('mean', ascending=False)
        fig = go.Figure(go.Bar(x=t.index, y=t['mean'] * 100, marker_color=BLUE,
                               text=[f'{v:.2%}' for v in t['mean']], textposition='outside',
                               customdata=t['size'], hovertemplate='%{x}<br>%{y:.2f}% · n=%{customdata:,}<extra></extra>'))
        fig.update_layout(title=col, yaxis_title='이상 비율 (%)')
        with box:
            show(style_fig(fig, height=320, legend=False))
    caveat('요일·시간대 차이는 Run 과 완전히 겹칩니다(confounded). 예: 토요일 = 01-08 GroupG Run 하루뿐 → 요일 효과로 해석 불가.')

# 5. 모델 ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader('정상여부 분류 — Run 단위 GroupKFold (Leave-one-run-out)')
    st.markdown('`정상여부` 는 센서값의 결정적 규칙이라 모델은 규칙을 다시 학습하는 셈입니다. '
                '목적은 **Run 이 바뀌어도 규칙이 일반화되는지**와 **판정에 쓰이는 센서** 확인입니다. '
                '같은 Run 이 학습·검증에 동시에 들어가지 않도록 11 fold(= Run 11개)로 나눕니다.')
    if st.button('모델 학습/결과 보기', type='primary') or st.session_state.get('furnace_model_ready'):
        st.session_state['furnace_model_ready'] = True
        res = fit_model()
        c = st.columns(3)
        c[0].metric('ROC-AUC', f"{res['auc']:.4f}")
        c[1].metric('PR-AUC', f"{res['ap']:.4f}", help=f"기준선(이상 비율) {res['base']:.4f}")
        c[2].metric('모델', 'HistGB · 200 iter')
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('**검증 Run 별 ROC-AUC**')
            st.dataframe(res['per_run'].style.format({'ROC-AUC': '{:.3f}', '이상 행': '{:,}'}, na_rep='- (이상 0건)'),
                         width='stretch')
        with c2:
            imp = res['importance']
            fig = go.Figure(go.Bar(x=imp.values, y=[short(i) for i in imp.index], orientation='h',
                                   marker_color=[BLUE if v > 0.01 else GRAY for v in imp.values],
                                   text=[f'{v:.3f}' for v in imp.values], textposition='outside'))
            fig.update_layout(title='Permutation importance (검증 run 7, PR-AUC 감소)', xaxis_title='PR-AUC 감소량')
            show(style_fig(fig, height=420, legend=False))
        caveat('규칙 기반 라벨이라 높은 AUC 는 당연한 결과입니다. 이상이 가장 많은 GroupG(run 7)는 AUC 0.926 으로 가장 낮아, '
               '학습에 없던 운전 조건에서는 판정이 흔들립니다.')
    else:
        st.caption('버튼을 누르면 전체 231,123행으로 11-fold 학습합니다 (최초 약 30초, 이후 캐시). '
                   '노트북 결과: ROC-AUC 0.9869 · PR-AUC 0.7447')

caveat('`ST` 는 평균 약 69 로 Z1_Temp(약 100.5)와 스케일이 달라 의미(설정값 여부)를 확인할 수 없습니다.')
