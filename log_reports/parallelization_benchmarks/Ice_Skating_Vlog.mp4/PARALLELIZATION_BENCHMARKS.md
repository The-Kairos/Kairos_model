# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-04-11 17:13:08 UTC | Ice_Skating_Vlog.mp4 | parallel | gemini | gemini-embedding-001 | 285.929 | 7.032 | 181.268 | 64.562 | 12.134 | 10.155 | 5.506 |

## 2026-04-11 17:13:08 UTC | Ice_Skating_Vlog.mp4 | parallel

- Video path: `/home/Kairos_model/.tmp/kairos/jobs/b035197d-be67-4606-b0ba-9adc524cca90/Ice_Skating_Vlog.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `285.929` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 7.032 |
| save_clips | - |
| sample_frames | 17.095 |
| caption_frames | 105.750 |
| sample_fps | 35.269 |
| detect_object_yolo | 48.542 |
| audio_scan | 31.984 |
| asr_timings | 46.028 |
| ast_timings | 149.274 |
| describe_scenes | 64.562 |
| summarize_scenes | 12.134 |
| synthesize_synopsis | 10.155 |
| make_embedding | 5.506 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 122.853 |
| branch_yolo_total | 83.820 |
| branch_audio_total | 181.268 |
