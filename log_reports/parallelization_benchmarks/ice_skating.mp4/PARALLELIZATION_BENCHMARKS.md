# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-04-10 21:15:36 UTC | Ice_Skating.mp4 | parallel | gemini | gemini-embedding-001 | 288.437 | 5.728 | 153.864 | 90.416 | 19.719 | 8.411 | 4.990 |

## 2026-04-10 21:15:36 UTC | Ice_Skating.mp4 | parallel

- Video path: `/home/Kairos_model/.tmp/kairos/jobs/be2e0309-fa15-458b-8055-42a4d5bdd80d/Ice_Skating.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `288.437` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 5.728 |
| save_clips | - |
| sample_frames | 18.102 |
| caption_frames | 104.112 |
| sample_fps | 24.496 |
| detect_object_yolo | 48.330 |
| audio_scan | 37.205 |
| asr_timings | 57.228 |
| ast_timings | 116.650 |
| describe_scenes | 90.416 |
| summarize_scenes | 19.719 |
| synthesize_synopsis | 8.411 |
| make_embedding | 4.990 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 122.221 |
| branch_yolo_total | 72.835 |
| branch_audio_total | 153.864 |
