# SceneWalk Held-Out Description Comparison

This note gives direct qualitative examples from the held-out SceneWalk evaluation. It compares SceneWalk ground-truth segment captions with Kairos descriptions generated using the frozen development-selected configuration.

## Frozen Configuration

- prompt set: `describe_scene_v4`, `fallback_describe_scene_v4`, `describe_scene_short_v4`
- aggregation: `fixed_window`
- aggregation window: `13s`
- aggregation max gap: `5s`
- held-out results file: `test/benchmarks/results/scenewalk_results_20260606_154048.json`
- comparison source file: `test/benchmarks/results/scenewalk_comparison.json`

## Held-Out Metrics

| Metric | Score |
|---|---:|
| Matched BERTScore F1 | `0.5886` |
| BERTScore Precision | `0.5885` |
| BERTScore Recall | `0.5903` |
| Matched ROUGE-L F1 | `0.2305` |
| SODA F1 | `0.1382` |
| SODA Precision | `0.1029` |
| SODA Recall | `0.2107` |
| Total matched pairs | `129` |

## Paper Narrative

The examples below should not be framed as SceneWalk being wrong or Kairos being automatically better. The more defensible claim is that automated overlap metrics can understate Kairos quality because the two systems describe different units of analysis. SceneWalk captions are broad segment-level references, while Kairos often produces more local, visually specific descriptions grounded in detected scenes and multimodal evidence.

This means no amount of ordinary prompt tuning or fixed temporal aggregation can fully eliminate the mismatch. Prompt tuning can improve style and grounding. Temporal aggregation can better align the scoring window. However, the reference caption and the Kairos description may still organize the same video evidence differently, include different visible details, and emphasize different temporal spans.

## Direct Examples

### Example 1: `c0VPJWt_f0w`, Kairos Scene `146`

- Kairos time: `1664.3s` to `1677.2s`
- SceneWalk reference time: `1664.0s` to `1697.7s`

**SceneWalk ground-truth description:**

```text
The video opens with a scene of a person sitting on a couch, followed by a close-up of a child's drawing. The next scene shows a person in a recording studio with audio equipment. Subsequently, a woman is seen entering a room, picking up a box, and walking out of the room. The video concludes with a scene of a person in a recording studio, surrounded by audio equipment.
```

**Kairos description:**

```text
An older woman and a child are seated together at a table, with the woman drawing on a piece of paper while the child watches closely. The next shot focuses on the drawing, which depicts two women sitting on a bench, held up by the child’s hand. The scene emphasizes the interaction between the woman, the child, and the drawing they are working on together. The scene takes place in a living room with two couches and a baby on the floor. A woman is initially seen lying on one of the couches near a window. The shot transitions to the same woman now sitting upright on the couch in the same living room. The space is defined by the couches, the window, and the presence of the baby on the floor. The scene begins with a close-up of a person’s hands holding a remote control, pressing buttons as if to play a video. The shot transitions to a man seated in a room, holding a video game controller and appearing to interact with a game. The next shot shows the same man playing a guitar in front of a monitor screen, with faint music audible in the background. The setting appears to be indoors, with the monitor and guitar as key objects defining the actions. The scene begins with a close-up of an old-style stereo system and various audio equipment, including a radio labeled "AVO's AV-705." The focus remains on the equipment, highlighting its analog design and components. The setting appears static, with no visible people or additional actions occurring in the frame.
```

**Why this matters:**

This example shows a reference-style mismatch: Kairos describes a local, visually specific unit, while SceneWalk describes a broader segment with a different narrative organization. Prompt tuning can make Kairos more concise, and temporal aggregation can align time windows, but neither fully removes the difference between local evidence-rich descriptions and broad human segment captions.

### Example 2: `c0VPJWt_f0w`, Kairos Scene `149`

- Kairos time: `1698.0s` to `1709.0s`
- SceneWalk reference time: `1697.7s` to `1721.2s`

**SceneWalk ground-truth description:**

```text
The video opens with a woman carrying a cardboard box with circular holes and a label that reads "KEEP DRY" and "CAUTION." The scene transitions to a close-up of a musical instrument's control panel with various knobs and a lit indicator. Next, a man is seen in a recording studio, singing into a microphone with a pop filter, surrounded by acoustic foam panels. The video then cuts to a child sitting on the floor, opening the cardboard box, and looking inside. The final scene returns to the man in the recording studio, now with his head down, seemingly in a state of distress or fatigue.
```

**Kairos description:**

```text
The scene begins in a dance studio where a man and a woman are dancing together, with their movements synchronized to the music. The shot transitions to an open box on the ground, labeled with the text "keep your home safe." The next frame shows a smaller box that has been removed from inside the larger one, placed nearby. The setting shifts focus entirely from the dancing to the boxes, with no other notable objects or actions visible. The scene focuses on a vintage analog mixing system with red indicator lights, buttons, and knobs visible on the control panel. The shot transitions to a close-up of a radio receiver with additional buttons and dials, emphasizing the detailed layout of the equipment. The final view highlights another section of the analog mixer, showcasing its intricate design of buttons and knobs. The setting appears static, with no visible people or movement in the frame. The scene takes place in an audio recording studio, focusing on various pieces of equipment. The first shot shows a control board with numerous buttons and sliders, viewed from the front. The next shot transitions to a recording machine featuring a screen with visible controls and buttons. Finally, the view shifts to an electronic music system with sound control components, emphasizing the studio's technical setup. No people are visible, and the focus remains on the audio equipment. The scene begins with a woman standing in front of a mirror, holding a microphone and speaking into it. She then picks up a cell phone and takes pictures of herself, still positioned near the mirror. The shot transitions to a television screen displaying a man in a green shirt speaking into a microphone. The setting alternates between the woman’s actions and the televised image, with the microphone and cell phone as key objects defining the interaction.
```

**Why this matters:**

This example shows a reference-style mismatch: Kairos describes a local, visually specific unit, while SceneWalk describes a broader segment with a different narrative organization. Prompt tuning can make Kairos more concise, and temporal aggregation can align time windows, but neither fully removes the difference between local evidence-rich descriptions and broad human segment captions.

### Example 3: `NkMWgw6hNrE`, Kairos Scene `70`

- Kairos time: `845.2s` to `857.7s`
- SceneWalk reference time: `826.5s` to `848.3s`

**SceneWalk ground-truth description:**

```text
The video opens with a close-up of a man with long hair and tattoos, driving a car. The scene shifts to a street where a man in a red shirt is seen running away from a car. The next scene shows the same man in the red shirt running through an alley with several others chasing him. The chase continues with the man in the red shirt running through the streets, with the pursuers closing in. The video concludes with the man in the red shirt getting into a car, with the pursuers still in close pursuit.
```

**Kairos description:**

```text
The scene takes place inside a car, where a woman is driving. The reflection of a man is visible, suggesting he is seated in the vehicle, though his exact position is unclear. The interior of the car is the primary visible setting, with no additional notable objects or actions occurring in the frame. The scene begins with a car driving in the rain, the image blurred by motion or weather effects. The shot transitions to a view of a telephone pole against a cloudy sky, with the word "omeio" faintly visible. It then shifts again to an electrical pole with the word "omelo" marked on it. The setting appears outdoors, with no other notable objects or actions visible beyond the stationary poles and the overcast atmosphere. The scene begins at night on a street where two men are visible playing guitars. The setting then shifts to a dark indoor room, where the same two men are present, and another person is seen walking across the floor. The shot transitions back to the street at night, where the two men are now running in opposite directions—one moving farther away while the other moves closer. The street is dimly lit, and no additional objects or landmarks are prominently visible. The scene begins indoors, showing a cluttered floor with debris and trash cans scattered around. It transitions to a nighttime street where several people are visible, including one person standing in the background. The focus shifts to a group of people playing on the street under artificial lighting, with a man standing prominently in front of them. The setting is defined by the urban environment, the nighttime atmosphere, and the interaction among the individuals.
```

**Why this matters:**

This example shows a reference-style mismatch: Kairos describes a local, visually specific unit, while SceneWalk describes a broader segment with a different narrative organization. Prompt tuning can make Kairos more concise, and temporal aggregation can align time windows, but neither fully removes the difference between local evidence-rich descriptions and broad human segment captions.

### Example 4: `NkMWgw6hNrE`, Kairos Scene `83`

- Kairos time: `983.8s` to `989.7s`
- SceneWalk reference time: `965.7s` to `989.5s`

**SceneWalk ground-truth description:**

```text
The video opens with a scene of two men facing each other in a confrontational stance, with one man appearing to be the aggressor and the other the target. The setting is a dimly lit, industrial-looking space with brick walls and visible pipes. The men are dressed in casual streetwear, with one wearing a white tank top and the other in a dark shirt. The confrontation escalates as the man in the white tank top appears to be the instigator, while the other man seems to be on the defensive.
In the subsequent scenes, the confrontation continues with various individuals joining or observing the altercation. The group's composition changes as more men enter the frame, some wearing similar casual clothing, while others are in different attire, such as a red sleeveless top. The atmosphere is tense, with physical contact between the men, suggesting a fight or scuffle. The lighting remains consistent throughout the scenes, with a focus on the central action.
The video concludes with a scene where the confrontation has escalated to a physical fight, with multiple men involved in the scuffle. The setting remains consistent, with the industrial backdrop and the group's casual attire. The altercation appears to be the focal point of the video, with the actions and interactions between the men being the primary narrative.
```

**Kairos description:**

```text
The scene takes place inside an urban building with graffiti-covered walls. Several people, all wearing masks, are visible in what appears to be an office-like space. Some individuals are moving around the room, with two people walking together toward the right while others shift positions or remain mostly stationary. The setting is defined by its graffiti-decorated walls and the casual movements of the group.
```

**Why this matters:**

This example shows a reference-style mismatch: Kairos describes a local, visually specific unit, while SceneWalk describes a broader segment with a different narrative organization. Prompt tuning can make Kairos more concise, and temporal aggregation can align time windows, but neither fully removes the difference between local evidence-rich descriptions and broad human segment captions.

## How To Use This In The Paper

- Include one or two examples in the qualitative/error-analysis section, not in the main metric table.
- Use the examples to explain why BERTScore, ROUGE-L, and SODA are useful but incomplete for evaluating detailed multimodal scene descriptions.
- State that the final held-out metrics are still reported transparently, while qualitative examples clarify the metric limitations.
- Avoid claiming that low overlap scores prove Kairos is better; claim that low overlap can reflect granularity and reference-style mismatch.
