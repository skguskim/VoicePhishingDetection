# Two-Stage Audio Processing Implementation Plan

## Goal
Split the audio processing workflow into two stages for consistency:
1.  **Stage 1 (Analysis)**: Calculate global statistics from the entire dataset to determine optimal processing parameters (e.g., global silence threshold).
2.  **Stage 2 (Processing)**: Apply these globally standardized parameters to process all audio files.

## Proposed Changes

### 1. [NEW] `analysis/analyze_audio_stats.py`
This script will scan the input directory and calculate global statistics.

*   **Logic**:
    *   Iterate through all audio files.
    *   Load each file (using `pydub` or `scipy`).
    *   Calculate `dBFS` (loudness) for each file.
    *   Collect all `dBFS` values.
    *   Calculate **Global Mean dB** across all files.
    *   Determine **Global Silence Threshold** based on `Global Mean dB + offset` (or a fixed percentile if requested, but mean+offset is standard).
    *   Save these values to `audio_processing_params.json`.

*   **Output JSON Format**:
    ```json
    {
      "global_mean_db": -25.5,
      "recommended_silence_thresh": -41.5,
      "analysis_file_count": 100
    }
    ```

### 2. [MODIFY] `analysis/process_audio.py`
Update the processing script to accept the JSON parameters file.

*   **Changes**:
    *   Add `--params_file` argument.
    *   If provided, read `silence_thresh` from the JSON file.
    *   If not provided, fall back to the old per-file calculation (but warn or prefer the new method).
    *   Use the loaded threshold strictly for `split_on_silence`.

## Verification Plan

### Manual Verification
1.  **Run Stage 1**:
    ```bash
    python analysis/analyze_audio_stats.py "C:\Path\To\Audio" --output_json "analysis/params.json"
    ```
    *   Check if `params.json` is created and contains reasonable values.

2.  **Run Stage 2**:
    ```bash
    python analysis/process_audio.py "C:\Path\To\Audio" --params_file "analysis/params.json"
    ```
    *   Check if files are processed.
    *   Verify that the split points seem consistent (no cut-off words due to overly aggressive adaptive thresholds).

### Resource Optimization
*   Ensure `tqdm` and `torch` (if GPU enabled) are used in both scripts where applicable to speed up loading/processing.
