# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-04-11 13:39:07 UTC | SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source__1___2__clip__7__clip_clip__2_.mp4 | parallel | gemini | gemini-embedding-001 | 30.641 | 0.312 | 9.857 | 5.788 | 2.450 | 6.678 | 0.839 |
| 2026-04-11 15:42:23 UTC | SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source__1___2__clip__7__clip_clip__2_.mp4 | parallel | gemini | gemini-embedding-001 | 25.938 | 0.075 | 6.264 | 5.177 | 1.574 | 7.686 | 0.739 |
| 2026-04-11 15:59:59 UTC | SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source__1___2__clip__7__clip_clip__2_.mp4 | parallel | gemini | gemini-embedding-001 | 28.463 | 0.078 | 6.235 | 5.898 | 2.211 | 8.841 | 0.712 |

## 2026-04-11 13:39:07 UTC | SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source__1___2__clip__7__clip_clip__2_.mp4 | parallel

- Video path: `/home/Kairos_model/.tmp/kairos/jobs/e94d313c-5a50-4d41-bfa3-2a2399e2baf5/SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source__1___2__clip__7__clip_clip__2_.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `30.641` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.312 |
| save_clips | - |
| sample_frames | 0.077 |
| caption_frames | 7.156 |
| sample_fps | 0.164 |
| detect_object_yolo | 3.908 |
| audio_scan | 5.729 |
| asr_timings | 2.966 |
| ast_timings | 4.118 |
| describe_scenes | 5.788 |
| summarize_scenes | 2.450 |
| synthesize_synopsis | 6.678 |
| make_embedding | 0.839 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 7.247 |
| branch_yolo_total | 4.097 |
| branch_audio_total | 9.857 |

## 2026-04-11 15:42:23 UTC | SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source__1___2__clip__7__clip_clip__2_.mp4 | parallel

- Video path: `/home/Kairos_model/.tmp/kairos/jobs/b0da3f3b-c39c-4e6a-94ae-8c4b6818ffca/SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source__1___2__clip__7__clip_clip__2_.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `25.938` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.075 |
| save_clips | - |
| sample_frames | 0.069 |
| caption_frames | 2.633 |
| sample_fps | 0.087 |
| detect_object_yolo | 0.908 |
| audio_scan | 2.105 |
| asr_timings | 2.345 |
| ast_timings | 4.149 |
| describe_scenes | 5.177 |
| summarize_scenes | 1.574 |
| synthesize_synopsis | 7.686 |
| make_embedding | 0.739 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 2.709 |
| branch_yolo_total | 1.003 |
| branch_audio_total | 6.264 |

## 2026-04-11 15:59:59 UTC | SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source__1___2__clip__7__clip_clip__2_.mp4 | parallel

- Video path: `/home/Kairos_model/.tmp/kairos/jobs/44c8220d-b382-49df-8c78-2b0e0dbadc0e/SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source__1___2__clip__7__clip_clip__2_.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `28.463` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.078 |
| save_clips | - |
| sample_frames | 0.076 |
| caption_frames | 2.745 |
| sample_fps | 0.067 |
| detect_object_yolo | 0.924 |
| audio_scan | 2.081 |
| asr_timings | 2.587 |
| ast_timings | 4.144 |
| describe_scenes | 5.898 |
| summarize_scenes | 2.211 |
| synthesize_synopsis | 8.841 |
| make_embedding | 0.712 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 2.831 |
| branch_yolo_total | 0.997 |
| branch_audio_total | 6.235 |
