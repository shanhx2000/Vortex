# Simulation run index

Newest last. One row per invocation of `run_simulation.sh` (named
`evaluation/run_all_simulation.sh` in the older records below).
Full detail, including output checksums, is in `runs/<id>/manifest.json`.

| Run id | Started | Duration | Jobs | -j | Status | Groups | Outputs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `20260806-204412_power_area` | 2026-08-06 20:44:12 | 58s | 1 | 1 | ok | power_area | vortex_power_area_breakdown.json |
| `20260806-204610_baseline_kernel+baseline_e2e_small` | 2026-08-06 20:46:10 | 112m | 61 | 32 | ok | baseline_kernel baseline_e2e_small | baseline_evaluation_results.csv, kernel_results.csv |
| `20260806-204612_kernel_eval+end_to_end_eval+impr_ablation+bat...` | 2026-08-06 20:46:12 | 346m | 693 | 16 | ok | kernel_eval end_to_end_eval impr_ablation batch_size_sweep power_area rebuttal | kernel_results.csv, rebuttal_batch_evaluation_results.csv, vortex_evaluation_results.csv, vortex_forceflow_evaluation_results.csv, vortex_power_area_breakdown.json, vortex_projection_attention_intensive_results.csv |
| `20260806-223753_baseline_e2e_all` | 2026-08-06 22:37:53 | 273m | 396 | 32 | ok | baseline_e2e_all | baseline_evaluation_results.csv |
| `20260808-165627_power_area+end_to_end_eval+impr_ablation+base...` | 2026-08-08 16:56:27 | 7m | 469 | 48 | ok | power_area end_to_end_eval impr_ablation baseline_e2e_small | baseline_evaluation_results.csv, vortex_evaluation_results.csv, vortex_power_area_breakdown.json, vortex_projection_attention_intensive_results.csv |
| `20260808-170611_kernel_eval` | 2026-08-08 17:06:11 | 4s | 1 | 1 | ok | kernel_eval | kernel_results.csv |
| `20260809-130855_baseline_sfu` | 2026-08-09 13:08:55 | 98m | 60 | 48 | ok | baseline_sfu | baseline_sfu_evaluation_results.csv |
| `20260809-193929_baseline_sfu_in_core` | 2026-08-09 19:39:29 | 127m | 60 | 48 | ok | baseline_sfu_in_core | baseline_sfu_in_core_evaluation_results.csv |
| `20260810-033432_power_area` | 2026-08-10 03:34:32 | 58s | 1 | 1 | ok | power_area | vortex_power_area_breakdown.json |
| `20260813-010703_baseline_kernel` | 2026-08-13 01:07:03 | 2s | 1 | 1 | ok | baseline_kernel | kernel_results.csv |
| `20260813-174019_baseline_kernel+kernel_eval+power_area` | 2026-08-13 17:40:19 | 58s | 3 | 1 | ok | baseline_kernel kernel_eval power_area | kernel_results.csv, vortex_power_area_breakdown.json |
| `20260813-174205_baseline_kernel+kernel_eval` | 2026-08-13 17:42:05 | 5s | 2 | 1 | ok | baseline_kernel kernel_eval | kernel_results.csv |
| `20260813-183322_baseline_kernel+kernel_eval` | 2026-08-13 18:33:22 | 5s | 2 | 1 | ok | baseline_kernel kernel_eval | kernel_results.csv |
