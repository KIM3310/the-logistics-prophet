# Review Guide - The Logistics Prophet

Updated: 2026-05-30

This repository is archived as a supporting proof. Review it for the reusable pattern, domain evidence, and portfolio relationship; do not treat it as the current flagship unless it is explicitly revived.

## Summary

| Field | Notes |
|---|---|
| Repository | `the-logistics-prophet` |
| Status | Archived supporting repository |
| Lane | Logistics prediction and operations control tower |
| Primary reader | 3PL operators, manufacturers, distribution networks, logistics SaaS teams, and operations consultancies. |
| Why it exists | Logistics teams need delay prediction, explainability, data quality, and operational action boards in one reviewable loop. |
| Stack | Python, Terraform, Docker |

## Open First

1. Read the README archived-status note and relationship to active repositories.
2. Inspect `docs/monetization-playbook.md` for the buyer lane and offer ladder.
3. Use the commands below to confirm the proof surface still has a review path.
4. Check CI workflows before making quality claims.
5. Keep the archived status visible in any portfolio conversation.

## Checks

| Purpose | Command |
|---|---|
| Full local gate | `make verify` |
| Test suite | `make test` |

## CI

- .github/workflows/architecture-blueprint.yml
- .github/workflows/ci.yml
- .github/workflows/dependency-review.yml
- .github/workflows/production-smoke.yml
- .github/workflows/repository-health.yml
- .github/workflows/repository-surface.yml
- .github/workflows/secret-scan.yml

## Evidence

- Synthetic data is clearly marked
- Evidence pack export works
- Optional integrations remain optional

## Commercial Notes

| Possible offer | Working price assumption | Scope |
|---|---|---|
| Control-tower assessment | $8k-$25k | Review current delay data, incident flow, and evidence-pack posture. |
| Lane-specific prediction pilot | $35k-$120k | Adapt the model, SHAP evidence, and operations console to one logistics lane. |
| Operations analytics retainer | $8k-$30k/month | Maintain quality gates, scorecards, and action-impact reviews. |

## Boundaries

- Do not claim production accuracy without customer data
- Keep operational recommendations human-reviewed
- Avoid implying live carrier integrations in static demos

## Useful Metrics

- Assessment starts
- Delay triage time saved
- Prediction calibration
- Retainer conversion
