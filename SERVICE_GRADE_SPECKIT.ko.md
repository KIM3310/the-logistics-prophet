# the-logistics-prophet Service-Grade SPECKIT

Last updated: 2026-03-08

## S - Scope
- 대상: predictive logistics control tower
- 이번 iteration 목표: pipeline health를 `control tower brief`로 끌어올려 모델/운영/거버넌스 언어를 한 화면에 묶기

## P - Product Thesis
- 이 repo는 예측 모델 데모가 아니라 `logistics control tower`로 읽혀야 한다.
- Palantir/Snowflake 스타일 대화에서 signal -> action mapping과 governance boundary가 동시에 보여야 한다.

## E - Execution
- service health report에 readiness contract, report schema, stages, operator rules를 추가
- Streamlit Control Tower에 `Control Tower Brief` + `Executive Review Pack` 패널을 상단 배치
- README에서 reviewer flow와 service-grade surface를 명시

## C - Criteria
- dashboard 실행 및 핵심 검증 green
- README 첫 부분에서 control-tower 가치가 설명됨
- 주요 signal과 action mapping이 흔들리지 않음
- health report와 dashboard copy가 같은 readiness 언어를 사용
- review pack이 queue parity, audit chain, approval gate를 reviewer 언어로 다시 묶는다

## K - Keep
- operations storytelling
- dashboard 중심 설계

## I - Improve
- scenario export / screenshot pack 정교화
- Snowflake / ontology narrative 확장
- multi-user review workflow surface

## T - Trace
- `src/control_tower/service_health.py`
- `app/dashboard.py`
- `tests/test_service_health.py`
- `README.md`
