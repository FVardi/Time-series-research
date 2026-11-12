# SimMTM

**Source:** Dong et al., NeurIPS 2023 (arXiv 2302.00861); official code github.com/thuml/SimMTM
(classification code), submodule `external/simmtm` at 169513b. Wrapped (D10 mode A): adapter
`harness/methods/simmtm.py` runs main.py's pretraining steps with the official `Load_Dataset`,
`TFC` model and `model_pretrain` per epoch (D109).
**Reported as:** SimMTM. **Settings:** `configs/methods/simmtm.yaml` (official SleepEEG config,
checked by the adapter; values changed for the benchmarks under `changed:`).

## Current rules
- main.py's argument defaults: lr 1e-4, 10 epochs, masking ratio 0.5, 3 masked copies, mean mask
  length 3, temperature 0.2 (D109, D145).
- Batch 128, the paper's classification value (code's SleepEEG config: 32) (D111).
- Multichannel: the encoder takes all channels and the decoder rebuilds all of them, as the
  official model does with several channels (D159).
- Per dataset (shape-forced): `TSlength_aligned` = window length, `CNNoutput_channel` (encoder
  output length; checked by the adapter), `input_channels` (D159, D173, D256).
- Run fixes: decoder size resized from the hard-coded SleepEEG Linear(1280, 178); trainer.py's
  import side effects undone (D109).

## Findings
- With two channels the official geometric mask is drawn over the interleaved values, so masked
  stretches per channel are shorter (2.25 vs 3 samples) (D159 note).
- 20 epochs gave no gain on FordA (D142, D145).
- XJTU-SY's 32,768-sample windows would need an 11 B-parameter decoder; no XJTU-SY config (D256).

History: D109, D111, D142, D145, D159, D173, D256.
