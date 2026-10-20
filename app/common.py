"""모든 페이지가 공유하는 경로·데이터 로딩·색상·차트 스타일."""
from pathlib import Path

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
    fig.update_xaxes(gridcolor=GRID, linecolor='#c3c2b7', zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor='#c3c2b7', zeroline=False)
    return fig


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
    desc = describe_table(df)
    fmt = {c: '{:,.4f}' for c in desc.columns}
    fmt.update({'count': '{:,.0f}', 'nunique': '{:,.0f}'})
    st.dataframe(desc.style.format(fmt), width='stretch')


def caveat(text: str) -> None:
    st.info(text, icon=':material/info:')
