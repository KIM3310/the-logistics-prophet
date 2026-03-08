# the-logistics-prophet Service-Grade SPECKIT

Last updated: 2026-03-08

## S - Scope
- 대상: predictive logistics control tower
- baseline 목표: demand/inventory/SLA signal을 운영 의사결정 surface로 고정

## P - Product Thesis
- 이 repo는 예측 모델 데모가 아니라 `logistics control tower`로 읽혀야 한다.
- Palantir/Snowflake 스타일 대화에서 signal -> action mapping이 보여야 한다.

## E - Execution
- KPI cards, forecast rationale, recommended actions를 같은 narrative로 정리
- sample scenarios와 dashboard proof를 유지
- build/test workflow를 운영 품질 기준으로 유지

## C - Criteria
- dashboard 실행 및 핵심 검증 green
- README 첫 부분에서 control-tower 가치가 설명됨
- 주요 signal과 action mapping이 흔들리지 않음

## K - Keep
- operations storytelling
- dashboard 중심 설계

## I - Improve
- scenario export / screenshot pack 정교화
- Snowflake / ontology narrative 확장

## T - Trace
- `README.md`
- `app/`
- `docs/`
- `.github/workflows/`

