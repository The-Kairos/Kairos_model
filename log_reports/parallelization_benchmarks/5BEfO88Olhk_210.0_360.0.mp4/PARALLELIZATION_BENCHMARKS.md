# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 17:30:20 UTC | 5BEfO88Olhk_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 139.982 | 0.793 | 50.866 | 15.543 | 8.756 | 12.611 | 3.359 |
| 2026-06-24 11:23:16 UTC | 5BEfO88Olhk_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 162.422 | 0.884 | 50.907 | 23.907 | 16.003 | 18.361 | 3.287 |

## 2026-06-23 17:30:20 UTC | 5BEfO88Olhk_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/5BEfO88Olhk_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `139.982` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.793 |
| save_clips | - |
| sample_frames | 0.978 |
| caption_frames | 35.258 |
| sample_fps | 2.174 |
| detect_object_yolo | 8.272 |
| audio_scan | 14.839 |
| asr_timings | 9.004 |
| ast_timings | 27.014 |
| describe_scenes | 15.543 |
| summarize_scenes | 8.756 |
| synthesize_synopsis | 12.611 |
| make_embedding | 3.359 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.242 |
| branch_yolo_total | 10.452 |
| branch_audio_total | 50.866 |

## 2026-06-24 11:23:16 UTC | 5BEfO88Olhk_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/5BEfO88Olhk_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `162.422` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.884 |
| save_clips | - |
| sample_frames | 0.959 |
| caption_frames | 36.196 |
| sample_fps | 2.209 |
| detect_object_yolo | 8.322 |
| audio_scan | 14.885 |
| asr_timings | 8.940 |
| ast_timings | 27.073 |
| describe_scenes | 23.907 |
| summarize_scenes | 16.003 |
| synthesize_synopsis | 18.361 |
| make_embedding | 3.287 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.160 |
| branch_yolo_total | 10.537 |
| branch_audio_total | 50.907 |
