# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-04-11 12:22:50 UTC | ice_skating.mp4 | parallel | gemini | gemini-embedding-001 | 298.968 | 5.807 | 190.321 | 65.375 | 19.032 | 8.340 | 4.729 |

## 2026-04-11 12:22:50 UTC | ice_skating.mp4 | parallel

- Video path: `/home/Kairos_model/.tmp/kairos/jobs/29f36b91-77ae-4c09-81e6-76e80913c3ab/ice_skating.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `298.968` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 5.807 |
| save_clips | - |
| sample_frames | 18.078 |
| caption_frames | 107.986 |
| sample_fps | 23.815 |
| detect_object_yolo | 49.064 |
| audio_scan | 37.001 |
| asr_timings | 52.541 |
| ast_timings | 153.307 |
| describe_scenes | 65.375 |
| summarize_scenes | 19.032 |
| synthesize_synopsis | 8.340 |
| make_embedding | 4.729 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 126.070 |
| branch_yolo_total | 72.887 |
| branch_audio_total | 190.321 |
