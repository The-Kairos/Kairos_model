# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 09:08:22 UTC | i_2ND2tpKAc_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 133.160 | 0.796 | 58.211 | 7.706 | 8.696 | 29.903 | 1.815 |

## 2026-06-26 09:08:22 UTC | i_2ND2tpKAc_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/i_2ND2tpKAc_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `133.160` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.796 |
| save_clips | - |
| sample_frames | 0.347 |
| caption_frames | 15.599 |
| sample_fps | 1.890 |
| detect_object_yolo | 6.712 |
| audio_scan | 12.011 |
| asr_timings | 35.932 |
| ast_timings | 10.259 |
| describe_scenes | 7.706 |
| summarize_scenes | 8.696 |
| synthesize_synopsis | 29.903 |
| make_embedding | 1.815 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 15.952 |
| branch_yolo_total | 8.609 |
| branch_audio_total | 58.211 |
