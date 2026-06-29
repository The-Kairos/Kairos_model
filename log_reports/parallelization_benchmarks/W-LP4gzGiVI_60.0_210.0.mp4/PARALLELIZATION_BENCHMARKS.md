# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 20:24:19 UTC | W-LP4gzGiVI_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 122.618 | 0.795 | 45.580 | 10.752 | 8.237 | 12.064 | 2.767 |

## 2026-06-25 20:24:19 UTC | W-LP4gzGiVI_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/W-LP4gzGiVI_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `122.618` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.795 |
| save_clips | - |
| sample_frames | 0.825 |
| caption_frames | 29.874 |
| sample_fps | 2.063 |
| detect_object_yolo | 8.182 |
| audio_scan | 10.840 |
| asr_timings | 13.035 |
| ast_timings | 21.696 |
| describe_scenes | 10.752 |
| summarize_scenes | 8.237 |
| synthesize_synopsis | 12.064 |
| make_embedding | 2.767 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.705 |
| branch_yolo_total | 10.251 |
| branch_audio_total | 45.580 |
