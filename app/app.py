"""실행 (테마 설정 .streamlit/config.toml 을 읽도록 프로젝트 루트에서):
    uv run streamlit run app/app.py
"""
import streamlit as st

st.set_page_config(page_title='1020_DS 공정 데이터 분석', page_icon=':material/monitoring:', layout='wide')

pages = {
    '개요': [
        st.Page('views/home.py', title='홈', icon=':material/home:', default=True),
    ],
    '반도체 공정': [
        st.Page('views/df_analysis.py', title='df_analysis — 불량 요인', icon=':material/memory:'),
        st.Page('views/df_semi_analysis.py', title='df_semi_analysis — 공정 단계 기여도', icon=':material/stacked_bar_chart:'),
    ],
    '다이캐스팅': [
        st.Page('views/diecasting.py', title='DieCasting — 불량 분석', icon=':material/precision_manufacturing:'),
    ],
    'Furnace': [
        st.Page('views/furnace_tube.py', title='Furnace_Tube_Data — 시계열 EDA', icon=':material/timeline:'),
        st.Page('views/furnace_analysis.py', title='Furnace_analysis — 이상치 라벨', icon=':material/warning:'),
    ],
}
st.navigation(pages).run()
