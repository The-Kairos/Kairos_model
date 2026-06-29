# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 03:46:08 UTC | HYO_eSo_Oow_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 225.652 | 0.783 | 82.251 | 17.407 | 10.031 | 12.698 | 6.872 |

## 2026-06-25 03:46:08 UTC | HYO_eSo_Oow_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/HYO_eSo_Oow_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `225.652` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.783 |
| save_clips | - |
| sample_frames | 1.739 |
| caption_frames | 75.961 |
| sample_fps | 2.713 |
| detect_object_yolo | 13.782 |
| audio_scan | 13.861 |
| asr_timings | 9.163 |
| ast_timings | 59.219 |
| describe_scenes | 17.407 |
| summarize_scenes | 10.031 |
| synthesize_synopsis | 12.698 |
| make_embedding | 6.872 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 77.706 |
| branch_yolo_total | 16.501 |
| branch_audio_total | 82.251 |
