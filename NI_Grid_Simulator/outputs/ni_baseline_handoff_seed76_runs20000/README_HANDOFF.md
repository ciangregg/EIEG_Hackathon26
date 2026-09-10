# Northern Ireland baseline handoff

This pack contains a frozen 20,000-scenario baseline.

## Contract for downstream optimisation
- Keep every frozen operating scenario unchanged.
- Keep `calibration_residual_mw` unchanged for each scenario.
- Change only the constraint-group policy/membership and the resulting network-group dispatch-down.
- A candidate is not an improvement if it makes a scenario insecure that is secure under the baseline.
- Report physical network/group dispatch-down separately from the calibrated total dispatch-down.

## Baseline headline metrics
- Physical model dispatch-down: 1.8878%
- Fixed calibration residual: 23.6122%
- Calibrated total dispatch-down: 25.5000%
- Calibrated renewable utilisation: 74.5000%
- Estimated transmission electrical efficiency: 98.8406%

The calibration residual is deliberately not an optimiser variable.
