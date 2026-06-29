# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-04-11 17:06:44 UTC | Argentina_v_France_Penalty_Shoot-out.mp4 | parallel | gemini | gemini-embedding-001 | 245.090 | 2.573 | 171.087 | 47.031 | 8.215 | 7.030 | 3.905 |

## 2026-04-11 17:06:44 UTC | Argentina_v_France_Penalty_Shoot-out.mp4 | parallel

- Video path: `/home/Kairos_model/.tmp/kairos/jobs/42fb0515-b3c1-48ba-8b80-a9365fe45efb/Argentina_v_France_Penalty_Shoot-out.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `245.090` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.573 |
| save_clips | - |
| sample_frames | 6.753 |
| caption_frames | 89.078 |
| sample_fps | 7.435 |
| detect_object_yolo | 37.189 |
| audio_scan | 42.908 |
| asr_timings | 25.111 |
| ast_timings | 128.169 |
| describe_scenes | 47.031 |
| summarize_scenes | 8.215 |
| synthesize_synopsis | 7.030 |
| make_embedding | 3.905 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 95.838 |
| branch_yolo_total | 44.634 |
| branch_audio_total | 171.087 |
