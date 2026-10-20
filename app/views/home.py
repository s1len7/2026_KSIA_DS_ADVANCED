import streamlit as st

st.title('반도체 Data Science 재직자 역량 강화 심화과정 2차')
st.caption('공정 데이터 분석 결과 대시보드  ·  분석 노트북은 `analysis/` 폴더 참고')

c1, c2, c3 = st.columns(3)
c1.metric('교육기간', '10/20 ~ 10/22', '총 21시간', delta_color='off')
c2.metric('교육장소', '경기도 교육센터', '반도체 아카데미', delta_color='off')
c3.metric('분석 데이터셋', '5개', '반도체 · 다이캐스팅 · Furnace', delta_color='off')
st.markdown('과정 안내: <https://academy.ksia.or.kr/lecture/offline/detail/747>')

st.divider()
st.subheader('데이터셋별 분석 페이지')

CARDS = [
    ('views/df_analysis.py', 'df_analysis', '1,664 웨이퍼 × 45 · CP949',
     'Thin F2·F4 가 핵심 인자. 산화온도 < 900°C & Thin F4 > 458 이면 불량률 100% (15/15).'),
    ('views/df_semi_analysis.py', 'df_semi_analysis', '1,664 웨이퍼 × 42',
     '같은 웨이퍼의 라벨 없는 버전. 식각 단계 정보만 다른 단계로 대체 불가.'),
    ('views/diecasting.py', 'DieCasting', '7,535 샷 × 18',
     '불량률 22.4%. 랜덤 CV 는 과대평가, 기간 효과가 지배적.'),
    ('views/furnace_tube.py', 'Furnace_Tube_Data', '231,123 행 × 15 · 2022-01-03 ~ 01-15',
     '11개 운전 run 구조. 튜브 온도 4개는 자기상관 ≈ 0.'),
    ('views/furnace_analysis.py', 'Furnace_analysis', '231,123 행 × 18',
     '정상여부=1 이 실제로는 "비정상". 전체 기준 IQR 라벨이라 run 차이가 이상으로 잡힘.'),
]
cols = st.columns(2)
for i, (page, name, shape, finding) in enumerate(CARDS):
    with cols[i % 2].container(border=True):
        st.markdown(f'**{name}**  \n:gray[{shape}]')
        st.write(finding)
        st.page_link(page, label='페이지 열기', icon=':material/arrow_forward:')

st.divider()
st.caption('모든 결과는 관측 데이터 기반 상관 분석이므로, 인과 확정에는 DOE(실험계획) 검증이 필요합니다.')
