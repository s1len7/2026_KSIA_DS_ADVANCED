# 반도체 Data Science 재직자 역량 강화 심화과정 2차

| 항목 | 내용 |
|---|---|
| 과정명 | 반도체 Data Science 재직자 역량 강화 심화과정 2차 |
| 사이트 | https://academy.ksia.or.kr/lecture/offline/detail/747 |
| 교육장소 | 반도체 아카데미 경기도 교육센터 |
<!-- | 교육기간 | 2026-10-20 (화) ~ 2026-10-22 (목), 총 21시간 | -->

반도체아카데미 재직자 DS 역량강화교육(심화과정)용 데이터 분석 작업 공간입니다.
공정 데이터(산화·포토·식각·이온주입·열처리, Furnace, Die Casting)를 탐색하고 불량 요인을 분석합니다.

## 폴더 구조

```
1020_DS/
├── analysis/                              # 분석 노트북 & 결과물 (데이터셋별 1개)
│   ├── df_analysis.ipynb                  # 반도체 공정 (라벨 포함본) 불량 요인 분석
│   ├── df_semi_analysis.ipynb             # 반도체 공정 (라벨 없는 본) Target 회귀·공정 단계 기여도
│   ├── DieCasting_analysis.ipynb          # 다이캐스팅 불량 분석
│   ├── Furnace_Tube_Data.ipynb            # Furnace 원본 시계열 EDA
│   ├── Furnace_analysis.ipynb             # Furnace 파생 컬럼(이상치 라벨) 분석
│   └── output/                            # 노트북별 요약 차트 (4 panel)
│       ├── analysis_summary.png           #   ← df_analysis
│       ├── df_semi_analysis_summary.png
│       ├── DieCasting_analysis_summary.png
│       ├── Furnace_Tube_Data_summary.png
│       └── Furnace_analysis_summary.png
├── app/                                   # Streamlit 웹 대시보드 (노트북과 별개로 CSV 에서 직접 계산)
│   ├── app.py                             # 진입점 (페이지 네비게이션)
│   ├── common.py                          # 공통: 데이터 로딩·캐시, 색상, 차트 스타일
│   ├── .streamlit/config.toml             # 테마 설정
│   └── views/                             # 페이지: home + 데이터셋별 5개
├── data/                                  # 원본 데이터 (수정하지 않음)
│   ├── df_analysis.csv
│   ├── df_semi_analysis.csv
│   ├── DieCasting_analysis.csv
│   ├── Furnace_analysis.csv
│   ├── Furnace_Tube_Data.csv
│   └── gpt_file/                          # 교육 자료 (안내문, 계획서, 발표자료, 예제 이미지)
├── backup/                                # 교육 원본 데이터셋 zip & 기존 실습 노트북 (01_Furnance, 02_CVD)
├── output/                                # 초기 분석 차트 (analysis/output/analysis_summary.png 와 동일)
├── .streamlit/config.toml                 # 테마 설정 (루트 실행·Streamlit Cloud 용)
├── pyproject.toml                         # 의존성 목록 (uv)
├── uv.lock                                # 설치 버전 고정 (uv 가 관리, 직접 수정하지 않음)
├── .python-version                        # Python 3.13
├── requirements.txt                       # uv.lock 에서 내보낸 앱 의존성 (Streamlit Cloud 용)
├── .venv/                                 # Python 가상환경 (uv 가 생성, git 제외)
└── python-manager-26.4.msix               # Python 설치 관리자
```

## 데이터셋

| 파일 | 크기 | 인코딩 | 내용 |
|---|---|---|---|
| `df_analysis.csv` | 1,664 × 45 | **CP949** | 웨이퍼별 공정 조건 + 품질 결과 (`Target`, `Defect`, `두께정상`, `CD정상`) |
| `df_semi_analysis.csv` | 1,664 × 42 | UTF-8 | `df_analysis` 와 같은 웨이퍼·같은 값, 파생 라벨 3개(`Defect`, `두께정상`, `CD정상`)만 없음 |
| `DieCasting_analysis.csv` | 7,535 × 18 | UTF-8 | 다이캐스팅 1샷 = 1행, 사출 속도·압력·온도·공장 환경 + `Defect` (시간 컬럼 없음) |
| `Furnace_Tube_Data.csv` | 231,123 × 15 | UTF-8 | 튜브 Furnace 센서 시계열 원본 (2022-01-03 ~ 01-15, 약 1초 간격, 결측 343셀) |
| `Furnace_analysis.csv` | 231,123 × 18 | UTF-8 | 위 원본의 결측을 평균 대체 + 파생 컬럼 3개(`이상치개수`, `정상여부`, `요일`) |

> `df_analysis.csv` 는 반드시 `pd.read_csv(..., encoding='cp949')` 로 읽어야 한글 컬럼이 깨지지 않습니다.

## 환경 설정

[uv](https://docs.astral.sh/uv/) 로 Python 3.13 가상환경(`.venv`)과 패키지를 관리합니다.
의존성은 `pyproject.toml` 에 있고, 설치되는 정확한 버전은 `uv.lock` 에 고정되어 있습니다.

- 앱 실행용: streamlit, pandas, numpy, plotly, scipy, scikit-learn
- 노트북용(dev 그룹): ipykernel, nbconvert, matplotlib, seaborn, statsmodels, xgboost, openpyxl

```powershell
# uv 설치 (한 번만)
winget install astral-sh.uv

# 환경 구성: Python 3.13 이 없으면 자동으로 받고, .venv 를 만들어 uv.lock 대로 설치
uv sync

# 패키지 추가 / 제거
uv add <패키지>            # 앱에 필요한 패키지
uv add --dev <패키지>      # 노트북에서만 쓰는 패키지
uv remove <패키지>

# 앱 의존성을 바꾼 뒤에는 Streamlit Cloud 용 requirements.txt 도 다시 생성
uv export --no-dev --no-hashes --no-emit-project -o requirements.txt
```

`uv run <명령>` 으로 실행하면 가상환경을 활성화하지 않아도 `.venv` 안에서 실행됩니다.
직접 활성화하려면 `.\.venv\Scripts\Activate.ps1`.

VS Code에서 노트북을 열 때는 커널로 **`.venv` (Python 3.13)** 를 선택하세요.
노트북은 `../data/` 상대경로로 데이터를 읽으므로 `analysis/` 폴더 위치를 유지해야 합니다.

## 분석 노트북 (Jupyter)

`analysis/` 폴더에 데이터셋별로 1개씩 있습니다. 분석 과정과 결과의 원본입니다.

### 작성 규칙

- `print()` 는 모두 f-string 사용, 문자열 안에 `\n` 을 쓰지 않고 줄바꿈은 `print()` 를 따로 호출
- 소수 값은 노트북·앱 모두 지수 표기 없이 천 단위 콤마 + 소수점 4자리(`{:,.4f}`)로 표시
  - 표·출력·차트 hover·막대 라벨 모두 해당. 아주 작은 p-value 도 `0.0000` 으로 표시
  - 예외: 퍼센트(`7.3%`)와 정수(개수)는 기존 형식 유지, 차트 축 눈금은 자동(지수 표기만 끔)
  - 표시 형식만 바꾸고 데이터 값은 반올림하지 않음 (노트북은 `pd.set_option('display.float_format', ...)`, 앱은 `common.fmt_table`·`style_fig`)
- 차트는 `analysis/output/<노트북명>_summary.png` 로 저장

### 재실행

```powershell
cd analysis
foreach ($nb in Get-ChildItem *.ipynb) {
    uv run jupyter nbconvert --to notebook --execute --inplace $nb.Name --ExecutePreprocessor.timeout=1800
}
```

### 1. `df_analysis.ipynb` — 반도체 공정 불량 요인

1. 데이터 점검 → 2. 라벨 정의 확인 → 3. 변수 정리 → 4. 단변량 분석 → 5. Target 회귀 → 6. Defect 분류 → 7. 규칙 추출 → 8. 고위험 조건·시각화

- **라벨 구조**: `Defect = Target ≥ 195`, `두께정상 = thickness ≥ 700`, `CD정상 = 22 ≤ Line_CD ≤ 55`
  → Defect 예측 시 `Target` 을 입력에 넣으면 leakage
- **불필요 컬럼**: `Flux840s`, `Temp_Oxid` (상수), `Vapor`(=`type`), `Wavelength`(=`UV_type`)
- **모델 성능**: Target 회귀 HistGB R² ≈ 0.73 · Defect 분류 RandomForest ROC-AUC ≈ 0.98
- **핵심 인자**: `Thin F2`, `Thin F4` ≫ `ppm`, `Thin F3`, `Temp_OXid`, `Pressure`
- **고위험 조건**

  | 조건 | 불량률 | n |
  |---|---|---|
  | 산화온도 < 900°C & Thin F4 > 458 | 100% | 15 |
  | Thin F4 > 458 & ppm ≤ 28.45 | 83% | 12 |
  | Thin F4 > 458 (그 외) | 21% | 164 |
  | 그 외 전체 | 2% | 1,029 |

- Dry 산화 불량률(9.5%)이 Wet(4.2%)의 2배 이상, Flux90s 비정상 저값 웨이퍼 23개의 불량률 43%

### 2. `df_semi_analysis.ipynb` — Target 회귀 & 공정 단계 기여도

`df_analysis` 와 동일한 웨이퍼라 결론이 겹치므로, **새로운 관점**(공정 단계별 기여도, 중간 품질지표) 위주로 분석했습니다.

- **데이터 관계**: 42개 공통 컬럼 100% 일치, 라벨 3개는 위 규칙으로 100% 재현됨
- **Target 회귀** (5-fold CV): HistGB R² 0.735 (MAE 17.2) > Poisson GLM 0.604 > Ridge 0.557
- **공정 단계별 기여도** (전체 모델 R² 0.732)

  | 단계 | 단독 R² | 제거 시 ΔR² |
  |---|---|---|
  | 식각 | 0.685 | −0.143 |
  | 포토 | 0.511 | −0.012 |
  | 이온주입 | 0.492 | +0.004 |
  | 산화 | 0.316 | −0.017 |
  | 열처리 | 0.117 | −0.000 |

  → **식각 단계 정보만 다른 단계로 대체 불가**, 이온주입·열처리는 고유 기여 없음
- **중간 품질지표**: `Line_CD` 는 포토 변수만으로 R² 0.751, `thickness` 는 산화 변수만으로 R² 0.372
- **고위험 조건** (중복 포함): 산화온도 < 900 & F4 > 458 → 15개 모두 Target ≥ 195,
  **F2 > 3666.5 & ppm ≤ 22.45 → 25개 중 84%** (신규), F4 > 458 & ppm ≤ 28.45 → 24개 중 91.7%

### 3. `DieCasting_analysis.ipynb` — 다이캐스팅 불량

- **라벨**: `Defect` 불량률 22.4% (1,689샷) · 제품별 Product 1 17.6% / Product 2 28.5%
- **데이터 품질**: `Velocity_1` 상수, 센서값 0 이상치 8행 제거(→ 7,527샷),
  직전 샷과 공정값이 완전히 같은 행 42.1%, Defect 자기상관(lag-1) 0.485
- **모델 성능 — 랜덤 CV는 과대평가**

  | 모델 | 랜덤 5-fold ROC-AUC | 시간 블록 5-fold ROC-AUC |
  |---|---|---|
  | Logistic | 0.718 | 0.676 |
  | RandomForest | 0.879 | 0.614 |
  | HistGB | 0.878 | 0.587 |

  → 인접 샷이 거의 같아서 랜덤 분할 시 정보가 새어 나감. **미래 구간 예측력은 약함** (시간 holdout AUC 0.553)
- **기간 효과가 지배적**: Product 1 불량률이 기간(Q1→Q4) 순으로 38.2% → 20.0% → 9.5% → 2.7%
- **같은 기간 안에서도 유지되는 규칙**
  - Product 2, `High_Velocity > 2.54` → 불량률 11.4% (n=754) vs 33.6% (Q1 안에서 7.6% vs 44.3%)
  - Product 1, `Biscuit_Thickness ≤ 11` → 불량률 24.5% (n=1,897) vs 11.9%
- 습도·냉각수 압력 관련 규칙은 특정 기간에 몰려 있어 기간 효과와 분리 불가

### 4. `Furnace_Tube_Data.ipynb` — Furnace 원본 시계열 EDA

- **기간·간격**: 2022-01-03 11:22 ~ 01-15 19:49, 분 단위 시각에 분당 약 58행 (≈ 1초 샘플링)
- **Run 구조**: `Code_Num` 이 바뀔 때마다 3.8시간 ~ 2일 공백 → **11개 운전 구간(run)**
  (GroupA~D 각 2회, E~G 각 1회). `Process_Type`(OP-A/B/C)은 행 단위로 23,360회 바뀌어 배치 구분이 아님
- **신호 특성**: OP·CP·ST·Z1 은 연속 제어 신호(자기상관 0.54~1.00),
  **튜브 온도 4개는 자기상관 ≈ 0** (885 ± 14 °C 의 독립 노이즈 형태)
- **이벤트**: GroupG run(01-08) 07시 이후 CP 0.455 → 0.11 급락, `Tube_Furnace1_OP` 결측이 run 2·10 에 집중

### 5. `Furnace_analysis.ipynb` — 파생 컬럼(이상치 라벨) 분석

- **원본과의 관계** (모두 100% 재현)
  - 결측 343셀 → 해당 컬럼 **전체 평균**으로 대체
  - `이상치개수` = 수치형 12개 컬럼 중 전체 데이터 기준 IQR 1.5배 범위를 벗어난 개수
  - `정상여부` = `이상치개수 ≥ 3` 이면 1 → **이름과 달리 1이 "비정상"** (3,940행, 1.70%)
  - `요일` = Datetime 의 요일
- **이상 판정 요인**: CP(이상 행의 66.0%), Z1_OP1, TF2_OP, TF3_OP, Z1_Temp 순 · 튜브 온도는 기여 0
- **run 별 이상률**: GroupG 15.80%, run 10 8.40%, run 1 7.60% … run 4·11 0%
- **라벨의 한계**: 범위를 전체 데이터로 잡아서 run 간 운전 조건 차이가 이상으로 잡힘.
  run 별로 다시 계산하면 이상률 1.70% → 0.73%, GroupG 는 15.80% → 0%
- **모델** (HistGB, run 단위 GroupKFold): ROC-AUC 0.987, PR-AUC 0.745 (base 0.017)
  → 라벨이 센서값에서 계산되는 규칙이라 높은 점수는 당연하며, run 간 일반화 확인용

> 모든 결과는 관측 데이터 기반 상관 분석이므로, 인과 확정에는 DOE(실험계획) 검증이 필요합니다.

---

## 웹 대시보드 (Streamlit)

노트북의 분석 결과를 웹페이지로 볼 수 있습니다. 노트북 파일은 사용하지 않고 `data/` 의 CSV 에서 직접 계산합니다.

### 실행

```powershell
cd C:\2026_KSIA_DS_ADVANCED
uv run streamlit run app/app.py
```

- 터미널에 `Local URL: http://localhost:8501` 이 뜨면 브라우저에서 접속합니다.
- **프로젝트 루트에서 실행**하세요. 테마 설정(`.streamlit/config.toml`)은 실행한 폴더에서 읽습니다. `app` 폴더에서 `uv run streamlit run app.py` 로 실행해도 됩니다(`app/.streamlit/` 사용).
- 다른 포트로 실행: `uv run streamlit run app/app.py --server.port 8502`
- 같은 네트워크의 다른 PC에서 보려면 터미널에 표시되는 `Network URL` 로 접속합니다 (Windows 방화벽 허용 필요).

### 종료

서버를 실행한 터미널에서 `Ctrl + C`.

### 화면 구성과 기본 조작

| 위치 | 기능 |
|---|---|
| 왼쪽 사이드바 | 페이지 이동 (개요 / 반도체 공정 / 다이캐스팅 / Furnace 그룹) |
| 페이지 상단 | 제목, 데이터·노트북 파일 경로, 핵심 지표(KPI) 카드 |
| 탭 | 분석 단계별 내용. 탭을 눌러 이동 |
| 슬라이더·선택 상자 | 값을 바꾸면 해당 차트·지표가 즉시 다시 계산됨 |
| 차트 | 마우스를 올리면 값 표시, 드래그로 확대, 더블클릭으로 원래 크기, 범례 클릭으로 항목 숨기기 |
| 표 | 열 제목 클릭으로 정렬, 오른쪽 위 아이콘으로 CSV 다운로드·검색·전체화면 |
| 파란 안내 상자 | 해석할 때 주의할 점 (leakage, 표본 수, 기간 효과 등) |
| 오른쪽 위 `⋮` 메뉴 | Rerun(새로고침), Clear cache(캐시 삭제), Settings(넓은 화면·테마) |

### 페이지별 사용 예

**홈** — 데이터셋별 요약 카드에서 `페이지 열기` 를 눌러 이동합니다.

**df_analysis — 불량 요인**
1. `영향 인자` 탭: `구간별 불량률을 볼 변수`(예: `Thin F2`)와 `구간 수 (분위)` 를 고르면 구간별 불량률이 표시됩니다. 점선은 전체 평균 7.3%.
2. `규칙·고위험 조건` 탭의 **조건 시뮬레이터**: `Thin F4 >`, `산화온도 <`, `ppm ≤` 슬라이더를 움직이면 조건에 해당하는 웨이퍼 수와 불량률(조건 안 / 밖)이 바로 바뀝니다.
   - 기본값(Thin F4 > 458, 산화온도 < 900) → 15개, 불량률 100%
   - 해당 웨이퍼가 30개 미만이면 표본이 적다는 경고가 나옵니다.
3. 오른쪽 `깊이` 를 바꾸면 Decision Tree 규칙이 다시 그려집니다.

**df_semi_analysis — 공정 단계 기여도**
1. `공정 단계 기여도` 탭: 단계별 단독 R² 와 제거 시 R² 감소량을 비교합니다.
2. `고위험 조건` 탭: Thin F2·F4·ppm·산화온도 슬라이더로 조건을 정하면 해당 웨이퍼 수, 평균 Target, Target ≥ 195 비율이 표시되고, 산점도에 해당 웨이퍼가 강조됩니다.

**DieCasting — 불량 분석**
1. `기간(시간) 효과` 탭: 제품과 변수를 고르면 이동평균 불량률과 변수 추이를 함께 볼 수 있습니다.
2. `모델` 탭: 랜덤 CV 와 시간 블록 CV 의 성능 차이를 확인합니다 (랜덤 CV 는 과대평가).
3. `규칙·고위험 조건` 탭: `추천 규칙` 에서 미리 정의된 규칙을 고르거나, 제품·변수·부등호·임계값을 직접 지정해 조건 안/밖 불량률을 비교합니다. 아래 차트에서 같은 기간 안에서도 차이가 유지되는지 확인하세요.

**Furnace_Tube_Data — 시계열 EDA**
1. `시계열 탐색` 탭: Run, 센서, 리샘플 간격(1분~1시간), 시간 범위를 고르면 해당 구간의 센서 추이가 표시됩니다. Run 사이 공백은 선이 끊겨 보입니다.
2. `이벤트·이상` 탭: GroupG run 의 CP 급락 구간과 Tube_Furnace1_OP 저출력 비율을 확인합니다.

**Furnace_analysis — 이상치 라벨**
1. 페이지 맨 위 안내: `정상여부 = 1` 이 **비정상** 입니다.
2. `이상 라벨` 탭: `IQR 배수 k` 와 `이상 판정 기준: 이탈 센서 수 ≥` 슬라이더를 바꾸면 이상 비율과 원본 라벨과의 일치율이 바뀝니다 (k=1.5, 3개일 때 원본과 100% 일치).
3. `Run별 이상률` 탭: 전체 기준 범위와 Run 별 범위로 판정했을 때의 차이를 비교합니다.
4. `모델` 탭: **`모델 학습/결과 보기` 버튼을 눌러야** 학습이 시작됩니다 (약 20초, 이후 캐시).

### 페이지 구성 요약

| 페이지 | 탭 | 조작 가능한 항목 |
|---|---|---|
| 홈 | 과정 정보, 데이터셋별 요약 카드 | 각 페이지로 이동 |
| df_analysis | 데이터 개요 · 라벨 구조 · 영향 인자 · 모델 성능 · 규칙·고위험 조건 | 변수·구간 수 선택, Thin F4 / 산화온도 / ppm 조건 시뮬레이터, 트리 깊이 |
| df_semi_analysis | 데이터 개요 · Target 분포 · 공정 단계 기여도 · 중간 품질지표 · 영향 인자·규칙 · 고위험 조건 | 구간 변수·개수, 트리 깊이, Thin F2·F4·ppm·산화온도 조건 필터 |
| DieCasting | 데이터 개요 · 제품별 불량률 · 기간 효과 · 랜덤 vs 시간 블록 CV · 영향 인자 · 규칙 | 제품·변수·부등호·임계값 선택, 기간 선택, 이동평균 창 |
| Furnace_Tube_Data | 데이터 개요 · Run 구조 · 시계열 탐색 · 신호 특성 · 이벤트·이상 | Run·센서·리샘플 간격·시간 범위 선택 |
| Furnace_analysis | 원본과의 관계 · 이상 라벨 · 이상 요인 · Run별 이상률 · 모델 | IQR 배수·이상 센서 개수 임계값, 모델 학습 버튼 |

### 참고 · 문제 해결

| 상황 | 원인 / 해결 |
|---|---|
| 페이지를 처음 열 때 20~30초 걸림 | 모델 교차검증을 계산하는 중입니다. 이후에는 캐시로 바로 열리며, 서버를 다시 시작하면 다시 계산합니다. |
| 오른쪽 위에 `Running...` 이 계속 표시됨 | 계산 중입니다. 끝날 때까지 기다리거나 `⋮` → `Rerun`. |
| CSV 를 바꿨는데 화면이 그대로임 | `⋮` → `Clear cache` 후 `Rerun` (키보드 `C` → `R`). |
| `Port 8501 is already in use` | 이미 서버가 실행 중입니다. 기존 터미널에서 `Ctrl + C` 로 끄거나 `--server.port 8502` 로 실행. |
| `streamlit` 명령을 찾을 수 없음 | 앞에 `uv run` 을 붙여 실행하세요: `uv run streamlit run app/app.py`. `uv` 도 없다고 나오면 터미널을 새로 열거나 uv 를 설치하세요. |
| `ModuleNotFoundError: common` | `views/` 안의 파일을 직접 실행한 경우입니다. 항상 `app.py` 로 실행하세요. |
| 노트북과 수치가 조금 다름 | 일부 페이지는 속도를 위해 트리 개수를 줄였습니다. 소수점 셋째 자리 수준의 차이입니다. |
