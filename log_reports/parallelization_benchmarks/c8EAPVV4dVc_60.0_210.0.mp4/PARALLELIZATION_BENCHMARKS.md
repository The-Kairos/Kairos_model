# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 02:06:32 UTC | c8EAPVV4dVc_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 176.752 | 0.819 | 61.836 | 16.839 | 16.192 | 11.301 | 4.473 |

## 2026-06-26 02:06:32 UTC | c8EAPVV4dVc_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/c8EAPVV4dVc_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `176.752` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.819 |
| save_clips | - |
| sample_frames | 1.466 |
| caption_frames | 49.816 |
| sample_fps | 2.402 |
| detect_object_yolo | 10.188 |
| audio_scan | 14.065 |
| asr_timings | 9.332 |
| ast_timings | 38.430 |
| describe_scenes | 16.839 |
| summarize_scenes | 16.192 |
| synthesize_synopsis | 11.301 |
| make_embedding | 4.473 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.287 |
| branch_yolo_total | 12.596 |
| branch_audio_total | 61.836 |
