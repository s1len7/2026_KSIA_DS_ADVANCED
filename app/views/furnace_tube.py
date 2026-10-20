"""Furnace_Tube_Data.csv — 튜브 Furnace 원본 시계열 EDA."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from common import (BLUE, GRAY, INK, MUTED, ORANGE, RED, SERIES, caveat, load_csv, page_header, show,
                    show_describe, style_fig)

NUM = ['Z1_Temp', 'Z1_OP1', 'ST',
       'Tube_Furnace1_OP', 'Tube_Furnace2_OP', 'Tube_Furnace3_OP', 'Tube_Furnace4_OP', 'Tube_Furnace_CP',
       'Tube_Furnace1_Temp', 'Tube_Furnace2_Temp', 'Tube_Furnace3_Temp', 'Tube_Furnace4_Temp']
TEMPS = [f'Tube_Furnace{i}_Temp' for i in range(1, 5)]
OPS = [f'Tube_Furnace{i}_OP' for i in range(1, 5)]
RESAMPLE = {'1분': '1min', '5분': '5min', '15분': '15min', '1시간': '1h'}


def short(name: str) -> str:
    return name.replace('Tube_Furnace', 'TF')


# ── 데이터 준비 (231k 행은 한 번만 읽고, 차트에는 집계값만 보냄) ──────────────────────
@st.cache_resource(show_spinner='Furnace_Tube_Data 불러오는 중...')
def get_data() -> pd.DataFrame:
    df = load_csv('Furnace_Tube_Data.csv').copy()
    df['dt'] = pd.to_datetime(df['Datetime'])
    df['run'] = (df['Code_Num'] != df['Code_Num'].shift()).cumsum()   # Code_Num 전환 = Run 경계
    return df


@st.cache_data
def run_table() -> pd.DataFrame:
    df = get_data()
    t = df.groupby('run').agg(Code_Num=('Code_Num', 'first'), start=('dt', 'min'), end=('dt', 'max'),
                              rows=('dt', 'size'))
    t['na_rows'] = df[NUM].isna().any(axis=1).groupby(df['run']).sum()
    t['minutes'] = ((t['end'] - t['start']).dt.total_seconds() / 60 + 1).astype(int)
    t['gap_before_h'] = ((t['start'] - t['end'].shift()).dt.total_seconds() / 3600).round(1)
    return t


@st.cache_data
def minute_means() -> pd.DataFrame:
    df = get_data()
    return df.groupby(['run', 'dt'])[NUM].mean().reset_index()


@st.cache_data
def signal_stats() -> pd.DataFrame:
    df = get_data()
    mm = df.groupby('dt')[NUM].mean()
    return pd.DataFrame({
        '행 단위 lag-1 자기상관': [df[c].autocorr(1) for c in NUM],
        '분 평균 lag-1 자기상관': [mm[c].autocorr(1) for c in NUM],
        '표준편차': [df[c].std() for c in NUM],
    }, index=NUM)


@st.cache_data
def zone_ranges() -> dict:
    df = get_data()
    mm = df.groupby('dt')[TEMPS].mean()
    row = (df[TEMPS].max(axis=1) - df[TEMPS].min(axis=1)).dropna()
    minute = mm.max(axis=1) - mm.min(axis=1)
    bins = np.arange(0, 64, 2)
    return {'bins': bins,
            'row_hist': np.histogram(row, bins)[0] / len(row) * 100,
            'min_hist': np.histogram(minute, bins)[0] / len(minute) * 100,
            'row_mean': row.mean(), 'min_mean': minute.mean(),
            'zone_mean': df[TEMPS].mean(),
            'run_zone': df.groupby('run')[TEMPS].mean()}


@st.cache_data
def op1_low_by_run(threshold: float) -> pd.Series:
    df = get_data()
    return (df['Tube_Furnace1_OP'] < threshold).groupby(df['run']).mean() * 100


# ── 페이지 ───────────────────────────────────────────────────────────────────────
page_header('Furnace_Tube_Data — 시계열 EDA',
            '튜브 Furnace 센서 원본 시계열: Run 구조, 신호/잡음, 구역 균일도, 이벤트',
            'Furnace_Tube_Data.csv', 'Furnace_Tube_Data.ipynb')

df = get_data()
runs = run_table()
mm = minute_means()

k = st.columns(5)
k[0].metric('행 수', f'{len(df):,}')
k[1].metric('기간', f"{df['dt'].min():%m-%d} ~ {df['dt'].max():%m-%d}", help=f"{df['dt'].min()} ~ {df['dt'].max()}")
k[2].metric('Run(가동 구간)', f'{len(runs)}개', help='Code_Num 이 바뀌는 지점 = 3.8시간~2일 공백')
k[3].metric('분당 행 수(중앙값)', f"{df.groupby('dt').size().median():.0f}", help='타임스탬프는 분 단위 — 약 1초 샘플링으로 추정')
k[4].metric('결측 셀', f"{int(df[NUM].isna().sum().sum()):,}")

tabs = st.tabs(['데이터 개요', 'Run 구조', '시계열 탐색', '신호 특성', '이벤트·이상'])

# 1. 데이터 개요 ---------------------------------------------------------------------
with tabs[0]:
    st.subheader('기술통계')
    show_describe(df[NUM])

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader('컬럼별 결측치')
        na = df[NUM].isna().sum()
        na = na[na > 0].sort_values()
        fig = go.Figure(go.Bar(x=na.values, y=[short(c) for c in na.index], orientation='h', marker_color=BLUE,
                               text=[f'{v:,}' for v in na.values], textposition='outside'))
        fig.update_layout(xaxis_title='결측 개수')
        show(style_fig(fig, height=320, legend=False))
    with c2:
        st.subheader('범주형 컬럼')
        pt = df['Process_Type'].value_counts()
        st.dataframe(pd.DataFrame({'행 수': pt, '비율(%)': (pt / len(df) * 100).round(2)}), width='stretch')
        switches = int((df['Process_Type'] != df['Process_Type'].shift()).sum())
        st.markdown(f'- `Process_Type` 전환 **{switches:,}회** → 배치 단위가 아니라 행 단위로 섞여 있음')
        st.markdown(f"- `Code_Num` {df['Code_Num'].nunique()}종 (GroupA~G)")
    caveat('`ST` 는 평균 약 69 로 Z1_Temp(약 100.5)·튜브 온도(약 885℃)와 스케일이 달라 의미(설정값 여부)를 확인할 수 없습니다.')

# 2. Run 구조 ------------------------------------------------------------------------
with tabs[1]:
    st.subheader('가동 구간(Run) 타임라인')
    fig = go.Figure()
    for r, row in runs.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['start'], row['end']], y=[f'run {r}'] * 2, mode='lines', line=dict(color=BLUE, width=14),
            hovertemplate=f"run {r} · {row['Code_Num']}<br>{row['start']:%m-%d %H:%M} ~ {row['end']:%m-%d %H:%M}"
                          f"<br>{row['rows']:,}행 · {row['minutes']}분<extra></extra>", showlegend=False))
        fig.add_annotation(x=row['end'], y=f'run {r}', text=f"  {row['Code_Num']}", showarrow=False,
                           xanchor='left', font=dict(size=11, color=MUTED))
    fig.update_yaxes(autorange='reversed')
    show(style_fig(fig, height=420, legend=False))

    tbl = runs.copy()
    tbl['start'] = tbl['start'].dt.strftime('%m-%d %H:%M')
    tbl['end'] = tbl['end'].dt.strftime('%m-%d %H:%M')
    st.dataframe(tbl.rename(columns={'rows': '행 수', 'na_rows': '결측 포함 행', 'minutes': '지속(분)',
                                     'gap_before_h': '직전 Run 과 공백(시간)'}),
                 width='stretch')

    st.subheader('Code_Num × Process_Type')
    st.dataframe(pd.crosstab(df['Code_Num'], df['Process_Type'], margins=True), width='stretch')

# 3. 시계열 탐색 ----------------------------------------------------------------------
with tabs[2]:
    c1, c2, c3 = st.columns([1.2, 2.6, 1])
    run_opts = ['전체'] + [f"run {r} · {c}" for r, c in runs['Code_Num'].items()]
    run_sel = c1.selectbox('Run', run_opts, index=7)
    sensors = c2.multiselect('센서', NUM, default=['Tube_Furnace_CP', 'Tube_Furnace1_OP', 'Tube_Furnace1_Temp'],
                             format_func=short)
    rule = c3.selectbox('집계 간격', list(RESAMPLE), index=0)

    sub = mm if run_sel == '전체' else mm[mm['run'] == int(run_sel.split()[1])]
    t0, t1 = sub['dt'].min().to_pydatetime(), sub['dt'].max().to_pydatetime()
    rng = st.slider('시간 범위', min_value=t0, max_value=t1, value=(t0, t1), format='MM-DD HH:mm')
    sub = sub[(sub['dt'] >= rng[0]) & (sub['dt'] <= rng[1])]

    if not sensors:
        st.warning('센서를 하나 이상 선택하세요.')
    elif sub.empty:
        st.warning('선택한 범위에 데이터가 없습니다.')
    else:
        # Run 별로 따로 리샘플한 뒤, Run 사이에는 빈 행을 넣어 선이 공백을 가로지르지 않게 함
        parts = []
        for _, g in sub.groupby('run'):
            rs = g.set_index('dt')[sensors].resample(RESAMPLE[rule]).mean().dropna(how='all')
            parts.append(rs)
            parts.append(pd.DataFrame(np.nan, index=[rs.index[-1] + pd.Timedelta(seconds=1)], columns=sensors))
        ts = pd.concat(parts)
        fig = make_subplots(rows=len(sensors), cols=1, shared_xaxes=True, vertical_spacing=0.03,
                            subplot_titles=[short(s) for s in sensors])
        for i, s in enumerate(sensors, start=1):
            fig.add_trace(go.Scatter(x=ts.index, y=ts[s], mode='lines', name=short(s),
                                     line=dict(color=SERIES[(i - 1) % len(SERIES)], width=1.6), connectgaps=False),
                          row=i, col=1)
        show(style_fig(fig, height=max(260, 170 * len(sensors)), legend=False))
        st.caption(f'{rule} 평균 · 표시 점 {ts.notna().any(axis=1).sum():,}개 (원본 행은 분당 약 58개) · 센서마다 y축을 따로 둠')

# 4. 신호 특성 -----------------------------------------------------------------------
with tabs[3]:
    stats = signal_stats()
    st.subheader('자기상관 — 연속 신호 vs 잡음')
    s = stats['행 단위 lag-1 자기상관'].sort_values()
    fig = go.Figure(go.Bar(x=s.values, y=[short(c) for c in s.index], orientation='h',
                           marker_color=[GRAY if v < 0.1 else BLUE for v in s.values],
                           text=[f'{v:.2f}' for v in s.values], textposition='outside'))
    fig.update_layout(xaxis_title='lag-1 자기상관 (행 단위)', xaxis_range=[0, 1.12])
    show(style_fig(fig, height=420, legend=False))
    st.dataframe(stats.style.format('{:,.4f}'), width='stretch')
    st.markdown('- OP·CP·ST·Z1 은 자기상관 0.54~1.00 → 연속적인 제어 신호\n'
                '- **튜브 온도 4개는 자기상관 ≈ 0** → 885℃ ± 14℃ 수준의 독립 잡음처럼 보임')

    st.subheader('튜브 구역 온도 균일도')
    z = zone_ranges()
    mid = (z['bins'][:-1] + z['bins'][1:]) / 2
    fig = go.Figure()
    fig.add_trace(go.Bar(x=mid, y=z['row_hist'], name=f"행 단위 (평균 {z['row_mean']:.1f}℃)", marker_color=GRAY))
    fig.add_trace(go.Bar(x=mid, y=z['min_hist'], name=f"분 평균 (평균 {z['min_mean']:.1f}℃)", marker_color=BLUE))
    fig.update_layout(barmode='overlay', xaxis_title='4개 구역 온도 편차 max-min (℃)', yaxis_title='비율 (%)')
    fig.update_traces(opacity=0.85)
    show(style_fig(fig, height=360))
    c1, c2 = st.columns([1, 2])
    c1.dataframe(z['zone_mean'].rename('평균 온도(℃)').round(2), width='stretch')
    c2.dataframe(z['run_zone'].round(2), width='stretch')

# 5. 이벤트·이상 ---------------------------------------------------------------------
with tabs[4]:
    st.subheader('GroupG Run(01-08) 후반 CP 급락')
    cp = mm[mm['run'] == 7].set_index('dt')['Tube_Furnace_CP']
    med = float(df['Tube_Furnace_CP'].median())
    fig = go.Figure(go.Scatter(x=cp.index, y=cp.values, mode='lines', line=dict(color=RED, width=2), name='CP 분 평균'))
    fig.add_hline(y=med, line=dict(color=INK, dash='dot', width=1), annotation_text=f'전체 중앙값 {med:.3f}',
                  annotation_position='top left')
    fig.update_layout(yaxis_title='Tube_Furnace_CP', yaxis_range=[0, 0.55])
    show(style_fig(fig, height=360, legend=False))
    st.markdown(f"- 07시 전까지 약 {cp.iloc[:120].mean():.3f} → 종료 직전 **{cp.iloc[-10:].mean():.3f}**")

    st.subheader('Tube_Furnace1_OP 저출력 구간')
    thr = st.slider('저출력 기준 (OP1 <)', 5.0, 60.0, 40.0, 1.0)
    low = op1_low_by_run(thr)
    fig = go.Figure(go.Bar(x=[f'run {r}' for r in low.index], y=low.values, marker_color=ORANGE,
                           text=[f'{v:.1f}%' for v in low.values], textposition='outside'))
    fig.update_layout(yaxis_title=f'OP1 < {thr:g} 비율 (%)')
    show(style_fig(fig, height=320, legend=False))
    st.markdown(f"- 전체 {(df['Tube_Furnace1_OP'] < thr).mean():.2%} 의 시간 동안 OP1 < {thr:g} (0 근처까지 떨어짐)")

    st.subheader('결측 블록')
    st.markdown('- `Tube_Furnace1_OP` 결측 254개는 **run 2(179행)·run 10(75행)** 에 연속 블록으로 몰려 있음 '
                '— 해당 구간 센서/수집 이상 가능성')
    caveat('이 페이지의 "이상"은 탐색적 관찰입니다. 라벨 기반 이상 판정(`이상치개수`, `정상여부`)은 '
           'Furnace_analysis 페이지에서 다룹니다.')
