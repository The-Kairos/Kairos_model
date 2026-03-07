# Retrieval Comparison: Watch Malala Yousafzai's Nobel Peace Prize acceptance speech.mp4

## Configuration

- **k** (top chunks): 5
- **top_c** (top clusters): 3
- **alpha** (cluster boost): 0.3
- **KMeans clusters**: 4
- **HDBSCAN clusters**: 0
- **Total queries**: 5

## Summary

| Method | Avg Time (s) | Avg Chunks | Avg Overlap vs Flat |
|--------|-------------:|-----------:|-------------------:|
| FLAT | 19.114 | 5.0 | 0.0% |
| KMEANS | 16.484 | 5.0 | 93.3% |
| HDBSCAN | 18.652 | 5.0 | 100.0% |

## Per-Query Results

## Query: Give me the clip of the woman with a colorful hijab sitting next to a man
### FLAT Retrieval
**Retrieved Chunks**:
1. [0.688] From 00:00:59.526 to 00:01:03.530, <short scene paragraph>  
The scene shows Malala Yousafzai, wearing a pink hijab, speaking at a podium during her Nobel Peace Prize acceptance speech. She appears to...
2. [0.675] From 00:01:11.271 to 00:01:18.478, <short scene paragraph>  
The scene shows Malala Yousafzai continuing her Nobel Peace Prize acceptance speech. She is seen wearing a red scarf and speaking at a podi...
3. [0.672] From 00:03:36.750 to 00:04:33.473, The scene shows Malala Yousafzai standing at a podium in a formal event setting, wearing pink, addressing an audience. She speaks about how people perceive her in di...
4. [0.667] From 00:03:30.210 to 00:03:33.413, <short scene paragraph>  
The scene shows two men in a room with flowers and a sign, sitting and talking to each other. The audio includes applause and a brief "Than...
5. [0.663] From 00:01:33.694 to 00:01:36.363, <short scene paragraph>  
The scene shows Malala Yousafzai, wearing a red scarf, speaking at a podium during her Nobel Peace Prize acceptance speech. The formal sett...

**Generated Answer**:
Not shown in the video.

**Timing**:
- Retrieval: 0.003s
- Generation: 18.186s
- Total: 18.189s

---

### KMEANS Retrieval
**Retrieved Chunks**:
1. [0.988] From 00:00:59.526 to 00:01:03.530, <short scene paragraph>  
The scene shows Malala Yousafzai, wearing a pink hijab, speaking at a podium during her Nobel Peace Prize acceptance speech. She appears to...
2. [0.975] From 00:01:11.271 to 00:01:18.478, <short scene paragraph>  
The scene shows Malala Yousafzai continuing her Nobel Peace Prize acceptance speech. She is seen wearing a red scarf and speaking at a podi...
3. [0.972] From 00:03:36.750 to 00:04:33.473, The scene shows Malala Yousafzai standing at a podium in a formal event setting, wearing pink, addressing an audience. She speaks about how people perceive her in di...
4. [0.964] From 00:03:30.210 to 00:03:33.413, <short scene paragraph>  
The scene shows two men in a room with flowers and a sign, sitting and talking to each other. The audio includes applause and a brief "Than...
5. [0.959] From 00:01:42.502 to 00:02:24.544, <short scene paragraph>  
The scene shows Malala Yousafzai speaking at a podium during her Nobel Peace Prize acceptance speech. She is wearing a red scarf, and the f...

**Generated Answer**:
Not shown in the video.

**Timing**:
- Retrieval: 0.004s
- Generation: 17.058s
- Total: 17.062s
- Overlap vs Flat: 66.7%

---

### HDBSCAN Retrieval
**Retrieved Chunks**:
1. [0.688] From 00:00:59.526 to 00:01:03.530, <short scene paragraph>  
The scene shows Malala Yousafzai, wearing a pink hijab, speaking at a podium during her Nobel Peace Prize acceptance speech. She appears to...
2. [0.675] From 00:01:11.271 to 00:01:18.478, <short scene paragraph>  
The scene shows Malala Yousafzai continuing her Nobel Peace Prize acceptance speech. She is seen wearing a red scarf and speaking at a podi...
3. [0.672] From 00:03:36.750 to 00:04:33.473, The scene shows Malala Yousafzai standing at a podium in a formal event setting, wearing pink, addressing an audience. She speaks about how people perceive her in di...
4. [0.667] From 00:03:30.210 to 00:03:33.413, <short scene paragraph>  
The scene shows two men in a room with flowers and a sign, sitting and talking to each other. The audio includes applause and a brief "Than...
5. [0.663] From 00:01:33.694 to 00:01:36.363, <short scene paragraph>  
The scene shows Malala Yousafzai, wearing a red scarf, speaking at a podium during her Nobel Peace Prize acceptance speech. The formal sett...

**Generated Answer**:
Not shown in the video.

**Timing**:
- Retrieval: 0.003s
- Generation: 16.289s
- Total: 16.292s
- Overlap vs Flat: 100.0%

---

## Query: Give me the clip of Kailash Satyarthi wearing glasses and white clothes clapping for Malala
### FLAT Retrieval
**Retrieved Chunks**:
1. [0.744] From 00:01:38.765 to 00:01:42.502, Scene Report:

The scene shows an audience clapping in a formal event setting, with applause and a brief "Thank you" heard in the audio. The visuals are inconsistent...
2. [0.744] From 00:03:27.674 to 00:03:30.210, The scene shows an audience clapping during a formal event, with applause and a brief "Thank you" heard in the audio. The visuals are inconsistent, with captions ref...
3. [0.743] From 00:01:36.363 to 00:01:38.765, <short scene paragraph>  
The scene shows an audience clapping and smiling during an event. The visuals suggest a formal setting, though there are inconsistencies in...
4. [0.742] From 00:03:24.337 to 00:03:27.674, <short scene paragraph>  
The scene shows an audience clapping during a formal event, with applause and a brief "Thank you" heard in the audio. The visuals are incon...
5. [0.741] From 00:00:21.755 to 00:00:25.893, Scene Report:

Characters:  
Name: Malala Yousafzai  
Role: Speaker, Nobel Peace Prize recipient  

Key Dialogue:  
Speaker: Malala Yousafzai  
Cause: Expressing gra...

**Generated Answer**:
Not shown in the video.

**Timing**:
- Retrieval: 0.004s
- Generation: 19.262s
- Total: 19.266s

---

### KMEANS Retrieval
**Retrieved Chunks**:
1. [1.044] From 00:01:38.765 to 00:01:42.502, Scene Report:

The scene shows an audience clapping in a formal event setting, with applause and a brief "Thank you" heard in the audio. The visuals are inconsistent...
2. [1.044] From 00:03:27.674 to 00:03:30.210, The scene shows an audience clapping during a formal event, with applause and a brief "Thank you" heard in the audio. The visuals are inconsistent, with captions ref...
3. [1.043] From 00:01:36.363 to 00:01:38.765, <short scene paragraph>  
The scene shows an audience clapping and smiling during an event. The visuals suggest a formal setting, though there are inconsistencies in...
4. [1.042] From 00:03:24.337 to 00:03:27.674, <short scene paragraph>  
The scene shows an audience clapping during a formal event, with applause and a brief "Thank you" heard in the audio. The visuals are incon...
5. [1.041] From 00:00:21.755 to 00:00:25.893, Scene Report:

Characters:  
Name: Malala Yousafzai  
Role: Speaker, Nobel Peace Prize recipient  

Key Dialogue:  
Speaker: Malala Yousafzai  
Cause: Expressing gra...

**Generated Answer**:
Not shown in the video.

**Timing**:
- Retrieval: 0.008s
- Generation: 19.851s
- Total: 19.859s
- Overlap vs Flat: 100.0%

---

### HDBSCAN Retrieval
**Retrieved Chunks**:
1. [0.744] From 00:01:38.765 to 00:01:42.502, Scene Report:

The scene shows an audience clapping in a formal event setting, with applause and a brief "Thank you" heard in the audio. The visuals are inconsistent...
2. [0.744] From 00:03:27.674 to 00:03:30.210, The scene shows an audience clapping during a formal event, with applause and a brief "Thank you" heard in the audio. The visuals are inconsistent, with captions ref...
3. [0.743] From 00:01:36.363 to 00:01:38.765, <short scene paragraph>  
The scene shows an audience clapping and smiling during an event. The visuals suggest a formal setting, though there are inconsistencies in...
4. [0.742] From 00:03:24.337 to 00:03:27.674, <short scene paragraph>  
The scene shows an audience clapping during a formal event, with applause and a brief "Thank you" heard in the audio. The visuals are incon...
5. [0.741] From 00:00:21.755 to 00:00:25.893, Scene Report:

Characters:  
Name: Malala Yousafzai  
Role: Speaker, Nobel Peace Prize recipient  

Key Dialogue:  
Speaker: Malala Yousafzai  
Cause: Expressing gra...

**Generated Answer**:
Not shown in the video.

**Timing**:
- Retrieval: 0.004s
- Generation: 20.450s
- Total: 20.454s
- Overlap vs Flat: 100.0%

---

## Query: Give me the clip of a room full of people clapping
### FLAT Retrieval
**Retrieved Chunks**:
1. [0.691] From 00:01:38.765 to 00:01:42.502, Scene Report:

The scene shows an audience clapping in a formal event setting, with applause and a brief "Thank you" heard in the audio. The visuals are inconsistent...
2. [0.689] From 00:01:36.363 to 00:01:38.765, <short scene paragraph>  
The scene shows an audience clapping and smiling during an event. The visuals suggest a formal setting, though there are inconsistencies in...
3. [0.683] From 00:03:27.674 to 00:03:30.210, The scene shows an audience clapping during a formal event, with applause and a brief "Thank you" heard in the audio. The visuals are inconsistent, with captions ref...
4. [0.680] From 00:03:24.337 to 00:03:27.674, <short scene paragraph>  
The scene shows an audience clapping during a formal event, with applause and a brief "Thank you" heard in the audio. The visuals are incon...
5. [0.672] From 00:03:33.413 to 00:03:36.750, <short scene paragraph>  
The scene shows a formal event with an audience clapping and a brief "Thank you" heard in the audio. The visuals inconsistently depict unre...

**Generated Answer**:
Here are the clips that show a room full of people clapping:

*   **00:01:36.363 to 00:01:38.765**: The scene shows an audience clapping and smiling during an event.
*   **00:01:38.765 to 00:01:42.502**: The scene shows an audience clapping in a formal event setting.
*   **00:03:24.337 to 00:03:27.674**: The scene shows an audience clapping during a formal event.
*   **00:03:27.674 to 00:03:30.210**: The scene shows an audience clapping during a formal event.
*   **00:03:33.413 to 00:03:36.750**: The scene shows a formal event with an audience clapping.

**Timing**:
- Retrieval: 0.003s
- Generation: 20.562s
- Total: 20.566s

---

### KMEANS Retrieval
**Retrieved Chunks**:
1. [0.991] From 00:01:38.765 to 00:01:42.502, Scene Report:

The scene shows an audience clapping in a formal event setting, with applause and a brief "Thank you" heard in the audio. The visuals are inconsistent...
2. [0.989] From 00:01:36.363 to 00:01:38.765, <short scene paragraph>  
The scene shows an audience clapping and smiling during an event. The visuals suggest a formal setting, though there are inconsistencies in...
3. [0.983] From 00:03:27.674 to 00:03:30.210, The scene shows an audience clapping during a formal event, with applause and a brief "Thank you" heard in the audio. The visuals are inconsistent, with captions ref...
4. [0.980] From 00:03:24.337 to 00:03:27.674, <short scene paragraph>  
The scene shows an audience clapping during a formal event, with applause and a brief "Thank you" heard in the audio. The visuals are incon...
5. [0.972] From 00:03:33.413 to 00:03:36.750, <short scene paragraph>  
The scene shows a formal event with an audience clapping and a brief "Thank you" heard in the audio. The visuals inconsistently depict unre...

**Generated Answer**:
Here are the clips of a room full of people clapping:
*   00:01:36.363 to 00:01:38.765
*   00:01:38.765 to 00:01:42.502
*   00:03:24.337 to 00:03:27.674
*   00:03:27.674 to 00:03:30.210
*   00:03:33.413 to 00:03:36.750

**Timing**:
- Retrieval: 0.004s
- Generation: 18.714s
- Total: 18.718s
- Overlap vs Flat: 100.0%

---

### HDBSCAN Retrieval
**Retrieved Chunks**:
1. [0.691] From 00:01:38.765 to 00:01:42.502, Scene Report:

The scene shows an audience clapping in a formal event setting, with applause and a brief "Thank you" heard in the audio. The visuals are inconsistent...
2. [0.689] From 00:01:36.363 to 00:01:38.765, <short scene paragraph>  
The scene shows an audience clapping and smiling during an event. The visuals suggest a formal setting, though there are inconsistencies in...
3. [0.683] From 00:03:27.674 to 00:03:30.210, The scene shows an audience clapping during a formal event, with applause and a brief "Thank you" heard in the audio. The visuals are inconsistent, with captions ref...
4. [0.680] From 00:03:24.337 to 00:03:27.674, <short scene paragraph>  
The scene shows an audience clapping during a formal event, with applause and a brief "Thank you" heard in the audio. The visuals are incon...
5. [0.672] From 00:03:33.413 to 00:03:36.750, <short scene paragraph>  
The scene shows a formal event with an audience clapping and a brief "Thank you" heard in the audio. The visuals inconsistently depict unre...

**Generated Answer**:
A room full of people can be seen clapping from 00:01:36.363 to 00:01:38.765. This action also occurs in the video from 00:01:38.765 to 00:01:42.502, 00:03:24.337 to 00:03:27.674, 00:03:27.674 to 00:03:30.210, and 00:03:33.413 to 00:03:36.750.

**Timing**:
- Retrieval: 0.003s
- Generation: 21.400s
- Total: 21.404s
- Overlap vs Flat: 100.0%

---

## Query: Where Malala says what her brothers call her
### FLAT Retrieval
**Retrieved Chunks**:
1. [0.725] From 00:03:36.750 to 00:04:33.473, The scene shows Malala Yousafzai standing at a podium in a formal event setting, wearing pink, addressing an audience. She speaks about how people perceive her in di...
2. [0.722] suggested_clips: 00:00:00 - Malala expresses pride as a Pashtun, Pakistani, and the youngest Nobel laureate. This moment sets the tone for her inspirational speech. | 00:00:28 - Malala humorously shar...
3. [0.711] highlights: 00:00:00 - Malala expresses pride in being the youngest Nobel laureate. | 00:00:28 - She humorously mentions still fighting with her brothers. | 00:01:42 - Malala describes herself as 'the...
4. [0.698] From 00:00:28.562 to 00:00:52.185, <short scene paragraph>  
Malala Yousafzai continues her Nobel Peace Prize acceptance speech, blending humor and humility as she reflects on her personal life and he...
5. [0.697] summary: The video mainly features Malala Yousafzai's Nobel Peace Prize acceptance speech, where she reflects on her journey, expresses gratitude, and advocates for education, peace, and equality. She...

**Generated Answer**:
From 00:03:36.750 to 00:04:33.473, Malala says, "my brother still called me that annoying bossy sister."

**Timing**:
- Retrieval: 0.005s
- Generation: 18.340s
- Total: 18.345s

---

### KMEANS Retrieval
**Retrieved Chunks**:
1. [1.022] suggested_clips: 00:00:00 - Malala expresses pride as a Pashtun, Pakistani, and the youngest Nobel laureate. This moment sets the tone for her inspirational speech. | 00:00:28 - Malala humorously shar...
2. [1.016] From 00:03:36.750 to 00:04:33.473, The scene shows Malala Yousafzai standing at a podium in a formal event setting, wearing pink, addressing an audience. She speaks about how people perceive her in di...
3. [1.011] highlights: 00:00:00 - Malala expresses pride in being the youngest Nobel laureate. | 00:00:28 - She humorously mentions still fighting with her brothers. | 00:01:42 - Malala describes herself as 'the...
4. [0.997] summary: The video mainly features Malala Yousafzai's Nobel Peace Prize acceptance speech, where she reflects on her journey, expresses gratitude, and advocates for education, peace, and equality. She...
5. [0.988] From 00:00:28.562 to 00:00:52.185, <short scene paragraph>  
Malala Yousafzai continues her Nobel Peace Prize acceptance speech, blending humor and humility as she reflects on her personal life and he...

**Generated Answer**:
At 00:03:36, Malala states, "my brother still called me that annoying bossy sister."

**Timing**:
- Retrieval: 0.006s
- Generation: 6.964s
- Total: 6.970s
- Overlap vs Flat: 100.0%

---

### HDBSCAN Retrieval
**Retrieved Chunks**:
1. [0.725] From 00:03:36.750 to 00:04:33.473, The scene shows Malala Yousafzai standing at a podium in a formal event setting, wearing pink, addressing an audience. She speaks about how people perceive her in di...
2. [0.722] suggested_clips: 00:00:00 - Malala expresses pride as a Pashtun, Pakistani, and the youngest Nobel laureate. This moment sets the tone for her inspirational speech. | 00:00:28 - Malala humorously shar...
3. [0.711] highlights: 00:00:00 - Malala expresses pride in being the youngest Nobel laureate. | 00:00:28 - She humorously mentions still fighting with her brothers. | 00:01:42 - Malala describes herself as 'the...
4. [0.698] From 00:00:28.562 to 00:00:52.185, <short scene paragraph>  
Malala Yousafzai continues her Nobel Peace Prize acceptance speech, blending humor and humility as she reflects on her personal life and he...
5. [0.697] summary: The video mainly features Malala Yousafzai's Nobel Peace Prize acceptance speech, where she reflects on her journey, expresses gratitude, and advocates for education, peace, and equality. She...

**Generated Answer**:
At 00:03:36, Malala says her brother calls her "that annoying bossy sister."

**Timing**:
- Retrieval: 0.009s
- Generation: 17.836s
- Total: 17.845s
- Overlap vs Flat: 100.0%

---

## Query: Give me the clip where Malala fixes her pink hijab
### FLAT Retrieval
**Retrieved Chunks**:
1. [0.740] From 00:00:59.526 to 00:01:03.530, <short scene paragraph>  
The scene shows Malala Yousafzai, wearing a pink hijab, speaking at a podium during her Nobel Peace Prize acceptance speech. She appears to...
2. [0.736] From 00:03:36.750 to 00:04:33.473, The scene shows Malala Yousafzai standing at a podium in a formal event setting, wearing pink, addressing an audience. She speaks about how people perceive her in di...
3. [0.728] suggested_clips: 00:00:00 - Malala expresses pride as a Pashtun, Pakistani, and the youngest Nobel laureate. This moment sets the tone for her inspirational speech. | 00:00:28 - Malala humorously shar...
4. [0.727] summary: The video mainly features Malala Yousafzai's Nobel Peace Prize acceptance speech, where she reflects on her journey, expresses gratitude, and advocates for education, peace, and equality. She...
5. [0.709] questions: Q: What is happening in the video? A: Malala Yousafzai delivers her Nobel Peace Prize acceptance speech, addressing themes of education, peace, and equality, interspersed with personal anec...

**Generated Answer**:
Not shown in the video.

**Timing**:
- Retrieval: 0.003s
- Generation: 19.201s
- Total: 19.204s

---

### KMEANS Retrieval
**Retrieved Chunks**:
1. [1.040] From 00:00:59.526 to 00:01:03.530, <short scene paragraph>  
The scene shows Malala Yousafzai, wearing a pink hijab, speaking at a podium during her Nobel Peace Prize acceptance speech. She appears to...
2. [1.035] From 00:03:36.750 to 00:04:33.473, The scene shows Malala Yousafzai standing at a podium in a formal event setting, wearing pink, addressing an audience. She speaks about how people perceive her in di...
3. [1.028] suggested_clips: 00:00:00 - Malala expresses pride as a Pashtun, Pakistani, and the youngest Nobel laureate. This moment sets the tone for her inspirational speech. | 00:00:28 - Malala humorously shar...
4. [1.027] summary: The video mainly features Malala Yousafzai's Nobel Peace Prize acceptance speech, where she reflects on her journey, expresses gratitude, and advocates for education, peace, and equality. She...
5. [1.009] questions: Q: What is happening in the video? A: Malala Yousafzai delivers her Nobel Peace Prize acceptance speech, addressing themes of education, peace, and equality, interspersed with personal anec...

**Generated Answer**:
Not shown in the video.

**Timing**:
- Retrieval: 0.009s
- Generation: 19.801s
- Total: 19.810s
- Overlap vs Flat: 100.0%

---

### HDBSCAN Retrieval
**Retrieved Chunks**:
1. [0.740] From 00:00:59.526 to 00:01:03.530, <short scene paragraph>  
The scene shows Malala Yousafzai, wearing a pink hijab, speaking at a podium during her Nobel Peace Prize acceptance speech. She appears to...
2. [0.736] From 00:03:36.750 to 00:04:33.473, The scene shows Malala Yousafzai standing at a podium in a formal event setting, wearing pink, addressing an audience. She speaks about how people perceive her in di...
3. [0.728] suggested_clips: 00:00:00 - Malala expresses pride as a Pashtun, Pakistani, and the youngest Nobel laureate. This moment sets the tone for her inspirational speech. | 00:00:28 - Malala humorously shar...
4. [0.727] summary: The video mainly features Malala Yousafzai's Nobel Peace Prize acceptance speech, where she reflects on her journey, expresses gratitude, and advocates for education, peace, and equality. She...
5. [0.709] questions: Q: What is happening in the video? A: Malala Yousafzai delivers her Nobel Peace Prize acceptance speech, addressing themes of education, peace, and equality, interspersed with personal anec...

**Generated Answer**:
Not shown in the video.

**Timing**:
- Retrieval: 0.003s
- Generation: 17.264s
- Total: 17.267s
- Overlap vs Flat: 100.0%

---

