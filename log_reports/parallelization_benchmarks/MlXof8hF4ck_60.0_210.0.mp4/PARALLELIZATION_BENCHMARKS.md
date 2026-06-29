# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 09:56:25 UTC | MlXof8hF4ck_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 191.398 | 0.783 | 62.308 | 16.176 | 24.866 | 24.975 | 3.923 |

## 2026-06-25 09:56:25 UTC | MlXof8hF4ck_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/MlXof8hF4ck_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `191.398` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.783 |
| save_clips | - |
| sample_frames | 1.151 |
| caption_frames | 44.232 |
| sample_fps | 2.302 |
| detect_object_yolo | 9.248 |
| audio_scan | 14.944 |
| asr_timings | 16.017 |
| ast_timings | 31.339 |
| describe_scenes | 16.176 |
| summarize_scenes | 24.866 |
| synthesize_synopsis | 24.975 |
| make_embedding | 3.923 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.389 |
| branch_yolo_total | 11.556 |
| branch_audio_total | 62.308 |
