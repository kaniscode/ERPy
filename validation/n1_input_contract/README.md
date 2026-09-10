# Checked hybrid input verification

The hybrid predictor requires the trained early-window inference configuration and matching source/trial provenance. `prepare_n1_hybrid_inputs` computes both paths from the same arrays and retains their separate finite masks. `FrozenN1Inputs` checks the immutable public feature and inference tables, their original configuration, exact record/epoch identities and recorded trial counts. Historical clean-trial indices were not archived and remain explicitly unavailable; no indices are invented.

The current interface replays 30 saved participant-held-out/site-transport models on 32,048 source-matched public records: all 128,192 calls are unchanged. All 64,096 stored participant-held-out scores agree within 4.44e-16. Transport tables store calls, not scores. Broad-window substitutions, including relabeled values, and bare inference dictionaries are rejected. No model was retrained, and frozen feature/model/prediction tables remain unchanged.

Verify the current source and evidence bindings:

```bash
python validation/verify_n1_input_contract.py --verify
```

Reexecute the stored-model comparisons into a new directory:

```bash
python validation/verify_n1_input_contract.py --output validation/work/n1_input_contract_replay
```

`verification.json` records exact input and executed-source hashes. These are prediction-interface checks, not a new estimate of physiological accuracy. See [N1 development](../../docs/N1_DEVELOPMENT.md) for the required configuration and [notebook 08](../../notebooks/examples/08_n1_development.ipynb) for a checked real-record example.
