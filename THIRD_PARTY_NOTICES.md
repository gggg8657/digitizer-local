# THIRD_PARTY_NOTICES — digitizer-local

이 저장소의 코드(`app.py`·`digitize.py`·`bars.py`·`synth.py`·`ui.html`·`selftest.py`·`setup.sh`)는 DongJu Kim(gggg8657)이 작성했으며 MIT 입니다. 제3자 코드·글꼴·아이콘은 동봉하지 않습니다. 아래는 설치해서 쓰거나, 외부 프로그램으로 부르거나, 문서에 실린 제3자 구성요소입니다.

조사 기준: 2026-10-12. 실제 파일(import 문, `requirements.txt`, `setup.sh`, `docs/img/`)을 보고, 라이선스는 패키지 메타데이터·원 저장소 LICENSE·논문 페이지에서 확인했습니다. 버전은 고정하지 않았습니다(`requirements.txt` 하한만).

## 구성요소

| 이름 | 버전 | 쓰임 | 라이선스 | 링크 | 상업 이용 | 비고 |
|---|---|---|---|---|---|---|
| NumPy | ≥1.21 | 설치(pip) | BSD-3-Clause | https://numpy.org | 가능 | |
| Pillow | ≥9.2 | 설치(pip) | MIT-CMU (HPND) | https://github.com/python-pillow/Pillow | 가능 | |
| openpyxl | ≥3.0 | 설치(pip, 선택 — XLSX 내보내기) | MIT | https://foss.heptapod.net/openpyxl/openpyxl | 가능 | |
| SciPy | — | 선택 import(있으면 산점 연결 성분 계산) | BSD-3-Clause | https://scipy.org | 가능 | requirements 에 없음 |
| matplotlib | — | 선택 import(`synth.py`·`selftest.py` 합성 그래프) | Matplotlib License (PSF 기반) | https://matplotlib.org | 가능 | requirements 에 없음 |
| poppler-utils `pdfinfo`·`pdftoppm` | — | 외부 프로그램(별도 프로세스 실행) | GPL-2.0 | https://poppler.freedesktop.org | 조건부 | 링크하지 않고 실행만 함. 동봉 안 함 |
| PyMuPDF (`fitz`) | — | 선택 import(poppler 가 없을 때만 PDF 폴백) | AGPL-3.0 또는 Artifex 상용 | https://github.com/pymupdf/PyMuPDF | **확인 필요** | requirements·setup 에 없음, 기본 미설치. [#1](https://github.com/gggg8657/digitizer-local/issues/1) |
| WebPlotDigitizer | — | 사용 흐름만 참고(코드 가져오지 않음) | AGPL-3.0 | https://github.com/automeris-io/WebPlotDigitizer | 해당 없음 | 축 두 점 보정 → 색 추출 → 수동 보정의 흐름만 참고 |
| Machala et al., "Life cycle comparison of industrial-scale lithium-ion battery recycling and mining supply chains", *Nat. Commun.* 16, 988 (2025), doi:10.1038/s41467-025-56063-x | — | 문서: README 예시의 입력 PDF, 스크린샷 `docs/img/upload.png`(쪽 썸네일)·`panels.png`·`overlay.png`(Fig. 6e 자르기 + 추출점 겹침)·`result.png`, 추출값 표·정확도 절의 Fig. 3·5·6 값 | CC BY 4.0 | https://creativecommons.org/licenses/by/4.0/ | 가능(출처 표기 조건) | PDF 원본은 저장소에 없음. 스크린샷은 원 그림을 자르고 보정 표시·추출점을 겹친 변형물. 추출값은 이 도구로 읽은 어림값이며 원 논문의 값이 아님 |
| Google Gemma 4 `gemma4:31b` (Ollama GGUF 빌드) | — | 외부 서비스(OpenAI 호환 API, 기본 VLM) | Apache-2.0 (Ollama `api/show` 의 license 필드로 확인) | https://ai.google.dev/gemma | 가능 | 가중치 동봉 안 함. `LLM_MODEL` 로 바꾸면 그 모델의 라이선스를 따름 |
| Ollama | — | 외부 서비스(LLM 서버) | MIT | https://github.com/ollama/ollama | 가능 | 동봉 안 함 |

## 확인 필요 (상업 이용 전)

- **PyMuPDF — AGPL-3.0** ([#1](https://github.com/gggg8657/digitizer-local/issues/1)): poppler-utils 가 없을 때만 쓰는 선택 폴백입니다. 상업 이용 시에는 PyMuPDF 를 설치하지 말고 `poppler-utils` 만 설치하세요(코드가 poppler 를 먼저 씁니다).
- **파동이(KAERI 캐릭터) — 옛 스크린샷 이력** ([#2](https://github.com/gggg8657/digitizer-local/issues/2)): 커밋 67694c6 이전의 `docs/img/*.png` 에 포털 펫 파동이가 찍혀 있었습니다. 지금 파일에는 없지만 공개 이력에 남아 있습니다. 캐릭터 권리는 KAERI·허쉬위쉬에 있습니다.

이 문서는 법률 자문이 아니며, 라이선스는 각 원문이 우선합니다.
