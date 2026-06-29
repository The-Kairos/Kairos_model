# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 20:08:41 UTC | C9JnjCVIW5k_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 145.723 | 0.833 | 46.348 | 13.095 | 7.812 | 16.223 | 3.890 |

## 2026-06-24 20:08:41 UTC | C9JnjCVIW5k_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/C9JnjCVIW5k_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `145.723` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.833 |
| save_clips | - |
| sample_frames | 1.253 |
| caption_frames | 43.307 |
| sample_fps | 2.357 |
| detect_object_yolo | 9.190 |
| audio_scan | 6.494 |
| asr_timings | 6.830 |
| ast_timings | 33.016 |
| describe_scenes | 13.095 |
| summarize_scenes | 7.812 |
| synthesize_synopsis | 16.223 |
| make_embedding | 3.890 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.566 |
| branch_yolo_total | 11.553 |
| branch_audio_total | 46.348 |
