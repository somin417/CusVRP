## Outputs Directory File Origin Mapping

이 문서는 `outputs/` 디렉토리에 있는 각 파일이 어떤 스크립트로부터 생성되었는지를 매핑합니다.

**디렉토리 구조:**
- `solutions/` - 모든 JSON solution 파일들
- `debug/` - 모든 debug JSON 파일들
- `plots/` - 모든 PNG 시각화 파일들
- `data/` - 모든 CSV 데이터 파일들
- `maps/` - HTML 맵 파일들
- `cache/` - 캐시 파일들
- `traces/` - Iteration trace CSV 파일들

## 실제 outputs 디렉토리 파일별 생성 스크립트

### JSON Solution 파일 (`outputs/solutions/`)

| 파일 | 생성 스크립트 | 함수/위치 |
|------|--------------|----------|
| `baseline.json` | `src/vrp_fairness/run_experiment.py` | `save_results()` 함수 |
| `local.json` | `src/vrp_fairness/run_experiment.py` | `save_results()` 함수 (local search algorithm, different from ALNS) |
| `ALNS_MAD.json` | `src/vrp_fairness/run_experiment.py` | `save_results()` 함수 (ALNS with MAD for Z3) |
| `ALNS_VAR.json` | `scripts/compare_Z3_variance_vs_MAD.py` | `main()` 함수 (ALNS with Variance for Z3, same experiment as ALNS_MAD) |
| `cts_solution.json` | `src/vrp_fairness/run_experiment.py` | `save_results()` 함수 (ALNS with CTS operator selection) |
| `solutions/abc_balance/cts_*.json` | `scripts/run_cts_abc_balance.py` | `save_best_solution()` 함수 |

### Debug JSON 파일 (`outputs/debug/`)

| 파일 | 생성 스크립트 | 함수/위치 |
|------|--------------|----------|
| `alns_mad_debug.json` | `src/vrp_fairness/run_experiment.py` | `main()` 함수 (ALNS with MAD) |
| `cts_debug.json` | `src/vrp_fairness/run_experiment.py` | `main()` 함수 |
| `variance_vs_mad_MAD_debug.json` | `scripts/compare_Z3_variance_vs_MAD.py` | `main()` 함수 |
| `variance_vs_mad_VARIANCE_debug.json` | `scripts/compare_Z3_variance_vs_MAD.py` | `main()` 함수 |
| `trace_analysis.json` | `scripts/utils/fix_best_solution_from_trace.py` | `main()` 함수 |

### CSV 데이터 파일 (`outputs/data/`)

| 파일 | 생성 스크립트 | 함수/위치 |
|------|--------------|----------|
| `baseline_metrics.csv` | `src/vrp_fairness/run_experiment.py` | `save_results()` 함수 |
| `baseline_vs_local_comparison.csv` | `src/vrp_fairness/run_experiment.py` | `save_results()` 함수 (local method) |
| `baseline_vs_alns_mad_comparison.csv` | `src/vrp_fairness/run_experiment.py` | `save_results()` 함수 (proposed method, fixed operator) |
| `baseline_vs_cts_comparison.csv` | `src/vrp_fairness/run_experiment.py` | `save_results()` 함수 (proposed method, cts operator) |
| `baseline_local_alns_mad_scores.csv` | `scripts/compare_waiting_and_scores.py` | `main()` 함수 |
| `baseline_local_alns_mad_metrics.csv` | `scripts/compare_waiting_and_scores.py` | `main()` 함수 |
| `baseline_local_alns_mad_wait_values.csv` | `scripts/compare_waiting_and_scores.py` | `main()` 함수 |
| `baseline_local_alns_mad_wait_hist_data.csv` | `scripts/compare_waiting_and_scores.py` | `main()` 함수 |
| `baseline_alns_variance_vs_mad_scores.csv` | `scripts/compare_Z3_variance_vs_MAD.py` | `main()` 함수 |
| `baseline_alns_variance_vs_mad_wait_values.csv` | `scripts/compare_Z3_variance_vs_MAD.py` | `main()` 함수 |
| `cts_vs_alns_scores.csv` | `scripts/compare_cts_vs_alns.py` | `main()` 함수 |
| `cts_vs_alns_wait_values.csv` | `scripts/compare_cts_vs_alns.py` | `main()` 함수 |
| `cts_vs_alns_wait_hist_data.csv` | `scripts/compare_cts_vs_alns.py` | `main()` 함수 |
| `abc_balance_summary.json` | `scripts/run_cts_abc_balance.py` | `main()` 함수 |
| `variance_vs_mad_results.json` | `scripts/compare_Z3_variance_vs_MAD.py` | `main()` 함수 |
| `weighted_waiting_graph_hist_data.csv` | `scripts/utils/generate_weighted_waiting_graph.py` | `main()` 함수 |
| `weighted_waiting_graph_values.csv` | `scripts/utils/generate_weighted_waiting_graph.py` | `main()` 함수 |
| `validation_comparison_metrics.csv` | `scripts/utils/validate_alns_cts_comparison.py` | `main()` 함수 |
| `traces/seed*_alns.csv` | `src/vrp_fairness/run_experiment.py` | `main()` 함수 (ALNS iteration trace) |
| `traces/seed*_cts.csv` | `src/vrp_fairness/run_experiment.py` | `main()` 함수 (CTS iteration trace) |

### PNG 시각화 파일 (`outputs/plots/`)

| 파일 | 생성 스크립트 | 함수/위치 |
|------|--------------|----------|
| `compare_wait_panels.png` | `scripts/compare_waiting_and_scores.py` | `plot_wait_panels()` 함수 |
| `waiting_plot.png` | `scripts/generate_waiting_plots_baseline_alns_cts.py` | `generate_waiting_plot()` 함수 |
| `weighted_waiting_plot.png` | `scripts/generate_waiting_plots_baseline_alns_cts.py` | `generate_weighted_waiting_plot()` 함수 |
| `waiting_times_z.png` | `scripts/generate_waiting_times_z.py` | `generate_waiting_plot()` 함수 |
| `weighted_waiting_times_z.png` | `scripts/generate_weighted_waiting_z.py` | `generate_weighted_waiting_plot()` 함수 |
| `abc_balance_comparison.png` | `scripts/run_cts_abc_balance.py` | `plot_comparison()` 함수 |
| `weighted_waiting_graph.png` | `scripts/utils/generate_weighted_waiting_graph.py` | `plot_weighted_waiting_histogram()` 함수 |
| `cts_vs_alns_wait_panels.png` | `scripts/compare_cts_vs_alns.py` | `plot_wait_panels()` 함수 |

### HTML 맵 파일 (`outputs/maps/`)

**참고:** Plot 관련 HTML 생성은 제거되었습니다. Map에서만 HTML이 생성됩니다.

| 파일 | 생성 스크립트 | 함수/위치 |
|------|--------------|----------|
| `map_compare.html` | `scripts/utils/generate_from_json.py` | `generate_map()` 함수 (--map 옵션) |
| `{run_id}_routes.html` | `src/vrp_fairness/run_experiment.py` | `main()` 함수 (실험 실행 시) |

### 캐시 파일 (`outputs/cache/`)

| 파일 | 생성 스크립트 | 함수/위치 |
|------|--------------|----------|
| `cache_osrm_polyline.json` | `src/vrp_fairness/inavi.py` | `iNaviCache` 클래스 |
| `cache_osrm_polyline.json.tmp` | `src/vrp_fairness/inavi.py` | `iNaviCache` 클래스 (임시 파일) |
| `cache_polyline.json` | `src/vrp_fairness/inavi.py` | `iNaviCache` 클래스 |
| `cache_inavi.json` | `src/vrp_fairness/inavi.py` | `iNaviCache` 클래스 |

### 로그 파일

| 파일 | 생성 스크립트 | 함수/위치 |
|------|--------------|----------|
| `z3_comparison.log` | `scripts/compare_Z3_variance_vs_MAD.py` | `main()` 함수 |

## 스크립트별 생성 파일 요약

### `compare_waiting_and_scores.py`
**직접 생성:**
- `data/baseline_local_alns_mad_scores.csv`
- `data/baseline_local_alns_mad_metrics.csv`
- `plots/compare_wait_panels.png`
- `data/baseline_local_alns_mad_wait_values.csv`
- `data/baseline_local_alns_mad_wait_hist_data.csv`

**간접 생성 (run_experiment.py subprocess 호출):**
- `solutions/baseline.json`
- `solutions/local.json`
- `solutions/ALNS_MAD.json`
- `debug/alns_mad_debug.json`
- `data/baseline_metrics.csv`
- `data/baseline_vs_alns_mad_comparison.csv`
- `solutions/seed*_alns.json`
- `traces/seed*_alns.csv`
- `solutions/seed*_alns_best.json`
- `maps/{run_id}_routes.html`

### `compare_cts_vs_alns.py`
**직접 생성:**
- `solutions/cts_solution.json`
- `debug/cts_debug.json`
- `data/cts_vs_alns_scores.csv`
- `plots/cts_vs_alns_wait_panels.png`
- `data/cts_vs_alns_wait_values.csv`
- `data/cts_vs_alns_wait_hist_data.csv`

**재사용:**
- `solutions/baseline.json` (compare_waiting_and_scores.py에서 생성)
- `solutions/ALNS_MAD.json` (compare_waiting_and_scores.py에서 생성)
- `debug/alns_mad_debug.json` (compare_waiting_and_scores.py에서 생성)

### `compare_Z3_variance_vs_MAD.py`
**직접 생성:**
- `data/variance_vs_mad_results.json`
- `debug/variance_vs_mad_MAD_debug.json`
- `debug/variance_vs_mad_VARIANCE_debug.json`
- `data/baseline_alns_variance_vs_mad_wait_values.csv` - Waiting time 값들 (plot 생성용)
- `data/baseline_alns_variance_vs_mad_scores.csv` - Z3 comparison scores
- `solutions/ALNS_VAR.json` (--reuse 모드가 아닐 때, ALNS with Variance for Z3)
- `z3_comparison.log`

**재사용/간접 생성:**
- `solutions/baseline.json` (내부에서 생성하거나 --reuse 모드에서 로드)
- `solutions/ALNS_MAD.json` (--reuse 모드에서 MAD solution으로 사용)

### `scripts/generate_waiting_plots_baseline_alns_cts.py`
**생성:**
- `plots/waiting_plot.png`
- `plots/weighted_waiting_plot.png`

**입력:**
- `solutions/baseline.json`
- `solutions/ALNS_MAD.json` (ALNS solution)
- `solutions/cts_solution.json`

### `scripts/generate_waiting_times_z.py`
**생성:**
- `plots/waiting_times_z.png`

**입력:**
- `solutions/baseline.json`
- `solutions/abc_balance/cts_z1_focused_best.json`
- `solutions/abc_balance/cts_balanced_best.json`
- `solutions/abc_balance/cts_z3_focused_best.json`

### `scripts/generate_weighted_waiting_z.py`
**생성:**
- `plots/weighted_waiting_times_z.png`

**입력:**
- `solutions/baseline.json`
- `solutions/abc_balance/cts_z1_focused_best.json`
- `solutions/abc_balance/cts_balanced_best.json`
- `solutions/abc_balance/cts_z3_focused_best.json`

### `scripts/run_cts_abc_balance.py`
**생성:**
- `plots/abc_balance_comparison.png` - Z scores comparison plot
- `data/abc_balance_summary.json` - Summary of all configurations
- `solutions/abc_balance/cts_z1_focused_best.json`
- `solutions/abc_balance/cts_balanced_best.json`
- `solutions/abc_balance/cts_z3_focused_best.json`

**입력:**
- `solutions/baseline.json` (from previous experiment, or specified via --baseline)

### `scripts/utils/generate_from_json.py`
**생성:**
- `plots/waiting_plot.png` (general utility, can generate from any JSON solutions)
- `plots/weighted_waiting_plot.png` (general utility)
- `maps/map_compare.html` (HTML은 map에서만 생성)

**입력:**
- `solutions/baseline.json`
- `solutions/ALNS_MAD.json` (or improved.json for backward compatibility)
- `solutions/local.json` (optional)
- `solutions/cts_solution.json` (optional)

### `scripts/utils/generate_map_from_compare.py`
**생성:**
- `map_compare_solutions.html`

**입력:**
- `baseline.json`
- `local.json` (optional)
- `ALNS_MAD.json` (optional, or improved.json for backward compatibility)
- `solutions/*_best.json` (optional, --include-best 플래그 사용 시)

### `scripts/utils/generate_weighted_waiting_graph.py`
**생성:**
- `plots/weighted_waiting_graph.png`
- `data/weighted_waiting_graph_hist_data.csv`
- `data/weighted_waiting_graph_values.csv`

**입력:**
- `data/baseline_local_alns_mad_wait_values.csv` (compare_waiting_and_scores.py에서 생성)
- `data/cts_vs_alns_wait_values.csv` (compare_cts_vs_alns.py에서 생성)
- `data/baseline_alns_variance_vs_mad_wait_values.csv` (compare_Z3_variance_vs_MAD.py에서 생성)

### `scripts/utils/regenerate_z3_plot.py`
**제거됨:** Plot 생성은 `generate_from_json.py`로 통합되었습니다.

### `scripts/utils/fix_best_solution_from_trace.py`
**생성:**
- `trace_analysis.json`

**입력:**
- `debug/alns_mad_debug.json`
- `solutions/ALNS_MAD.json`
- `solutions/baseline.json`

## 파일 의존성 그래프

```
compare_waiting_and_scores.py
  ├─> run_experiment.py (subprocess)
  │     ├─> baseline.json
  │     ├─> local.json
  │     ├─> ALNS_MAD.json
  │     ├─> debug/alns_mad_debug.json
  │     ├─> baseline_metrics.csv
  │     ├─> baseline_vs_alns_mad_comparison.csv
  │     ├─> solutions/seed*_alns.json
  │     ├─> traces/seed*_alns.csv
  │     ├─> solutions/seed*_alns_best.json
  │     └─> maps/{run_id}_routes.html
  └─> baseline_local_alns_mad_scores.csv
      ├─> baseline_local_alns_mad_metrics.csv
      ├─> compare_wait_panels.png
      ├─> baseline_local_alns_mad_wait_values.csv
      └─> baseline_local_alns_mad_wait_hist_data.csv

compare_cts_vs_alns.py
  ├─> (재사용) baseline.json, ALNS_MAD.json, alns_mad_debug.json
  └─> cts_vs_alns_scores.csv
      ├─> cts_vs_alns_wait_panels.png
      └─> cts_vs_alns_wait_values.csv

compare_Z3_variance_vs_MAD.py
  ├─> variance_vs_mad_results.json
  ├─> variance_vs_mad_MAD_debug.json
  ├─> variance_vs_mad_VARIANCE_debug.json
  ├─> baseline_alns_variance_vs_mad_wait_values.csv
  └─> z3_comparison.log

generate_weighted_waiting_graph.py
  ├─> (입력) baseline_local_alns_mad_wait_values.csv
  ├─> (입력) cts_vs_alns_wait_values.csv
  ├─> (입력) baseline_alns_variance_vs_mad_wait_values.csv
  └─> weighted_waiting_graph.png
      ├─> weighted_waiting_graph_hist_data.csv
      └─> weighted_waiting_graph_values.csv

generate_waiting_plots_baseline_alns_cts.py
  ├─> (입력) baseline.json, ALNS_MAD.json, cts_solution.json
  ├─> waiting_times_baseline_alns_cts.png
  └─> weighted_waiting_times_baseline_alns_cts.png

generate_waiting_times_z.py
  ├─> (입력) baseline.json, best_solutions/abc_balance/*.json
  └─> waiting_times_abc_balance_models.png

generate_weighted_waiting_z.py
  ├─> (입력) baseline.json, best_solutions/abc_balance/*.json
  └─> weighted_waiting_times_abc_balance_models.png

run_cts_abc_balance.py
  ├─> (입력) baseline.json
  ├─> abc_balance_comparison.png
  ├─> abc_balance_summary.json
  └─> best_solutions/abc_balance/*.json

generate_from_json.py
  ├─> (입력) baseline.json, ALNS_MAD.json, local.json, cts_solution.json 등
  ├─> waiting_plot.png (--waiting-plot 옵션, general utility)
  ├─> weighted_waiting_plot.png (--weighted-plot 옵션, general utility)
  └─> map_compare.html (--map 옵션, HTML은 map에서만 생성)
```

## 파일 생성 순서 권장사항

1. **기본 실험 실행:**
   ```bash
   python scripts/compare_waiting_and_scores.py
   ```
   → `baseline.json`, `local.json`, `ALNS_MAD.json`, `baseline_local_alns_mad_*.csv`, `compare_wait_panels.png` 생성

2. **CTS 비교 (선택):**
   ```bash
   python scripts/compare_cts_vs_alns.py
   ```
   → `cts_vs_alns_*.csv`, `cts_vs_alns_*.png` 생성

3. **Z3 Variance vs MAD 비교 (선택):**
   ```bash
   python scripts/compare_Z3_variance_vs_MAD.py
   ```
   → `variance_vs_mad_*.json`, `variance_vs_mad_*.png` 생성

4. **Plot/Map 생성:**
   ```bash
   python scripts/utils/generate_from_json.py --baseline outputs/solutions/baseline.json --improved outputs/solutions/ALNS_MAD.json
   python scripts/utils/generate_weighted_waiting_graph.py
   ```
