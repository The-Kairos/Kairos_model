# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 18:36:00 UTC | U2Svw1Y9X3E_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 135.226 | 0.917 | 50.843 | 10.680 | 8.867 | 13.331 | 3.077 |

## 2026-06-25 18:36:00 UTC | U2Svw1Y9X3E_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/U2Svw1Y9X3E_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `135.226` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.917 |
| save_clips | - |
| sample_frames | 1.203 |
| caption_frames | 34.567 |
| sample_fps | 2.309 |
| detect_object_yolo | 7.952 |
| audio_scan | 16.420 |
| asr_timings | 9.596 |
| ast_timings | 24.819 |
| describe_scenes | 10.680 |
| summarize_scenes | 8.867 |
| synthesize_synopsis | 13.331 |
| make_embedding | 3.077 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.776 |
| branch_yolo_total | 10.267 |
| branch_audio_total | 50.843 |
