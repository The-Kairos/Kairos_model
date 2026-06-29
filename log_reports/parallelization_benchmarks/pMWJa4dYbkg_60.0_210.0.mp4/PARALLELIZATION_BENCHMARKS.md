# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 08:05:55 UTC | pMWJa4dYbkg_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 116.669 | 0.772 | 45.436 | 9.975 | 7.675 | 12.618 | 2.499 |

## 2026-06-28 08:05:55 UTC | pMWJa4dYbkg_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/pMWJa4dYbkg_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `116.669` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.772 |
| save_clips | - |
| sample_frames | 0.763 |
| caption_frames | 25.832 |
| sample_fps | 2.061 |
| detect_object_yolo | 7.642 |
| audio_scan | 16.031 |
| asr_timings | 11.007 |
| ast_timings | 18.389 |
| describe_scenes | 9.975 |
| summarize_scenes | 7.675 |
| synthesize_synopsis | 12.618 |
| make_embedding | 2.499 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 26.601 |
| branch_yolo_total | 9.709 |
| branch_audio_total | 45.436 |
