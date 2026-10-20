"""모든 페이지가 공유하는 경로·데이터 로딩·색상·차트 스타일."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'

# 색상 (노트북 차트와 동일)
BLUE, ORANGE, RED, GRAY, INK = '#2a78d6', '#eb6834', '#e34948', '#b5b3ab', '#0b0b0b'
AQUA, YELLOW, VIOLET = '#1baf7a', '#eda100', '#6250d6'
SERIES = [BLUE, ORANGE, AQUA, YELLOW, '#e87ba4', '#008300', VIOLET, RED]
MUTED, GRID, SURFACE = '#52514e', '#ecebe7', '#fcfcfb'

# 소수 표시 형식: 지수 표기 없이 천 단위 콤마 + 소수점 4자리 (값은 그대로, 표시만)
FLOAT_FMT = '{:,.4f}'


@st.cache_data(show_spinner='데이터 불러오는 중...')
def load_csv(name: str, encoding: str = 'utf-8', **kwargs) -> pd.DataFrame:
    return pd.read_csv(DATA / name, encoding=encoding, low_memory=False, **kwargs)


def page_header(title: str, caption: str, source: str, notebook: str) -> None:
    st.title(title)
    st.caption(f'{caption}  ·  데이터 `data/{source}`  ·  노트북 `analysis/{notebook}`')


def style_fig(fig: go.Figure, height: int = 380, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(family='Pretendard, "Malgun Gothic", sans-serif', size=13, color=INK),
        title_font=dict(size=15),
        plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
        showlegend=legend,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0, title=None),
        hoverlabel=dict(bgcolor='white', font_size=12),
        bargap=0.25,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor='#c3c2b7', zeroline=False,
                     exponentformat='none', separatethousands=True)
    fig.update_yaxes(gridcolor=GRID, linecolor='#c3c2b7', zeroline=False,
                     exponentformat='none', separatethousands=True)
    _set_hoverformat(fig)
    return fig


def _axis_values(fig: go.Figure, letter: str, axis_id: str) -> list:
    """해당 축에 그려지는 trace 값 (histogram 의 개수 축은 정수로 간주)."""
    vals = []
    for tr in fig.data:
        if (getattr(tr, f'{letter}axis', None) or letter) != axis_id:
            continue
        v = getattr(tr, letter, None)
        if v is None:
            if tr.type == 'histogram':
                vals.append(0)
            continue
        vals.extend(np.asarray(v).ravel().tolist())
    return vals


def _set_hoverformat(fig: go.Figure) -> None:
    """hover 값: 실수 축은 #,##0.0000, 정수 축은 #,##0. 퍼센트(tickformat %)·범주·날짜 축은 그대로."""
    for ax_name in [k for k in fig.layout.to_plotly_json() if k.startswith(('xaxis', 'yaxis'))]:
        ax = fig.layout[ax_name]
        if ax.hoverformat or '%' in (ax.tickformat or '') or ax.type in ('category', 'date', 'multicategory'):
            continue
        letter = ax_name[0]
        vals = _axis_values(fig, letter, letter + ax_name[5:])
        nums = [v for v in vals if isinstance(v, (int, float, np.number)) and not isinstance(v, bool)]
        if not nums or len(nums) < len(vals):     # 문자열·날짜가 섞이면 범주/날짜 축
            continue
        finite = [v for v in nums if np.isfinite(v)]
        ax.hoverformat = ',d' if finite and all(float(v).is_integer() for v in finite) else ',.4f'


def fmt_table(data: pd.DataFrame | pd.Series, formats: dict | None = None):
    """st.dataframe 용 Styler: 실수 컬럼은 #,##0.0000, formats 로 컬럼별 예외 지정 (퍼센트 등)."""
    df = data.to_frame() if isinstance(data, pd.Series) else data
    floats = [c for c in df.columns if pd.api.types.is_float_dtype(df[c])]
    sty = df.style.format(FLOAT_FMT, subset=floats, na_rep='')
    if pd.api.types.is_float_dtype(df.index):
        sty = sty.format_index(FLOAT_FMT)
    return sty.format(formats) if formats else sty


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width='stretch', theme=None)


def describe_table(df: pd.DataFrame) -> pd.DataFrame:
    """노트북과 같은 기술통계 + nunique + 변동계수."""
    desc = df.describe().T
    desc['count'] = desc['count'].astype(int)
    desc['nunique'] = df[desc.index].nunique()
    desc['cv%'] = desc['std'] / desc['mean'] * 100
    return desc


def show_describe(df: pd.DataFrame) -> None:
    """지수 표기 없이 천 단위 콤마 + 소수점 4자리."""
    st.dataframe(fmt_table(describe_table(df)), width='stretch')


def caveat(text: str) -> None:
    st.info(text, icon=':material/info:')
