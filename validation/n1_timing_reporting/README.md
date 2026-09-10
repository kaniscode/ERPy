# N1 timing reporting verification

Current feature extraction and portable model predictions distinguish measured N1 timing from the model's unchanged 50-ms midpoint imputation. `measured_peak_latency_ms` is null when there is no interior negative peak; `peak_latency_imputed` flags the imputed feature input. Unavailable features have no measured timing and are not classified. The original `peak_latency_ms` feature remains unchanged to preserve trained models.

`verification.json` records a current-source replay of all 26 saved participant-held-out models over 32,048 paired records: 64,096 individual predictions. All calls match and score differences are below 1e-12. Each of the 10,648 no-peak model predictions has unavailable measured timing and an explicit imputation flag. The 5,324 no-peak records contain 76 hybrid calls and 15.5 positive reference weight, giving subgroup PPV 20.39%. This does not establish timing accuracy or justify a new hard peak gate.

The receipt identifies the fixed inputs, models and historical source identities
by SHA-256. These are prediction-interface checks, not a new performance study.
Notebook 08 checks its model inputs and compares its prediction with the saved
participant-held-out result before displaying the measured timing fields.
