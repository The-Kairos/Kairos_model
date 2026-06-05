# TIB Benchmark Report — Kairos

**Dataset:** TIB (TIB AV-Portal academic video archive)
**Date:** 2026-06-05
**Videos:** 10 long-form presentations (60–113 min each)
**Task:** Compare Kairos full-video synopsis against human-written video abstracts

---

## Aggregate Metrics

| Metric              |   Score |
|---------------------|---------|
| BERTScore F1        |  0.5931 |
| BERTScore Precision |  0.6415 |
| BERTScore Recall    |  0.5537 |
| ROUGE-L F1          |  0.1163 |
| ROUGE-L Precision   |  0.2367 |
| ROUGE-L Recall      |  0.0892 |
| BLEU-1              |  0.0797 |
| BLEU-2              |  0.0295 |
| BLEU-3              |  0.0053 |
| BLEU-4              |  0.0014 |

---

## Per-Video Breakdown

| # | Title | Duration | GT Words | Gen Words | BERTScore F1 | ROUGE-L F1 | BLEU-1 |
|---|-------|----------|----------|-----------|-------------|-----------|--------|
| 1 | Building Custom Pinball Machines | 63.0 min | 33 | 36 | 0.6684 | 0.1690 | 0.2632 |
| 2 | Differential Privacy and the US Census | 70.1 min | 151 | 37 | 0.5999 | 0.1036 | 0.0215 |
| 3 | Lecture 03. Reactions of Organometallic Reagents | 80.3 min | 76 | 33 | 0.6274 | 0.1296 | 0.0764 |
| 4 | General Relativity \| Lecture 3 | 112.7 min | 48 | 36 | 0.5948 | 0.1395 | 0.1883 |
| 5 | Panel - Ask the EFF: Digital Civil Liberties | 107.9 min | 720 | 41 | 0.5229 | 0.0412 | 0.0000 |
| 6 | Snails & Hawkwings | 73.8 min | 53 | 39 | 0.6030 | 0.1474 | 0.1466 |
| 7 | Insecure coding in C (and C++) | 63.4 min | 92 | 31 | 0.6174 | 0.1760 | 0.0507 |
| 8 | "The" Social Credit System | 61.1 min | 153 | 32 | 0.5586 | 0.0632 | 0.0064 |
| 9 | Automotive Ethernet PHY bring-up | 60.4 min | 103 | 29 | 0.5873 | 0.1212 | 0.0405 |
| 10 | Microsoft Azure Web Jobs | 67.1 min | 136 | 26 | 0.5508 | 0.0727 | 0.0036 |

---

## Ground Truth vs Generated Previews

### 1. Building Custom Pinball Machines (63.0 min)

**Ground Truth:**
> How to build a pinball machine? We introduce you to all basics and explain the different options for hardware and software. As an example, we show images of our own custom pinball machine.

**Kairos Generated:**
> A computer scientist shares the journey of designing and building a custom two-player pinball machine, covering game design, mechanics, electronics, software, iterative problem-solving, and lessons learned.

---

### 2. Differential Privacy and the US Census (70.1 min)

**Ground Truth:**
> Differential privacy is a mathematically rigorous definition of privacy tailored to statistical analysis of large datasets. Differentially private systems simultaneously provide useful statistics to the analyst and strong protection to the individuals in the data...

**Kairos Generated:**
> The video explores the introduction, applications, and challenges of differential privacy in the U.S. Census, emphasizing its role in protecting individual data while maintaining statistical utility...

---

### 3. Lecture 03. Reactions of Organometallic Reagents (80.3 min)

**Ground Truth:**
> UCI Chem 51C Organic Chemistry (Spring 2012) Lec 03. Organic Chemistry - Reactions of Organometallic Reagents - Instructor: James S. Nowick, Ph.D. This is the third quarter course in the organic chemistry...

**Kairos Generated:**
> A detailed university lecture explores organometallic reagents, retrosynthetic analysis, reaction mechanisms, and practical applications, emphasizing nucleophilic behavior, stereochemical outcomes, and safety...

---

### 4. General Relativity | Lecture 3 (112.7 min)

**Ground Truth:**
> (October 8, 2012) Leonard Susskind continues his discussion of Riemannian geometry and uses it as a foundation for general relativity. This series is the fourth installment of a six-quarter series that...

**Kairos Generated:**
> The lecture explores the mathematical foundations of General Relativity, focusing on curvature tensors, Christoffel symbols, and Riemannian geometry, with the professor using humor, demonstrations, and...

---

### 5. Panel - Ask the EFF: Digital Civil Liberties (107.9 min)

**Ground Truth:**
> Ask the EFF: The Year in Digital Civil Liberties — Get the latest information about how the law is racing to catch up with technological change from staffers at the Electronic Frontier Foundation...

**Kairos Generated:**
> The DEF CON panel by the Electronic Frontier Foundation (EFF) covered digital civil liberties, NSA surveillance, global privacy advocacy, whistleblower protections, encryption, legislative reforms, and...

---

### 6. Snails & Hawkwings (73.8 min)

**Ground Truth:**
> This lecture is about his fascination with Ferdinando Sanfelice, the Napolitan nobleman and architect of crazy, absolutely beautiful, staircases. It was given at the EPFL...

**Kairos Generated:**
> Fabrizio Balabino delivers an in-depth lecture on Ferdinando San Felice's architectural innovations, focusing on Baroque staircase designs, their adaptation to Naples' dense urban fabric, and the integration...

---

### 7. Insecure coding in C (and C++) (63.4 min)

**Ground Truth:**
> Let's turn the table. Suppose your goal is to deliberately create buggy programs in C and C++ with serious security vulnerabilities that can be "easily" exploited...

**Kairos Generated:**
> The speaker humorously explains insecure coding practices in C and C++, covering vulnerabilities like buffer overflows and ROP, protection mechanisms like ASLR and DEP, and exploitation techniques...

---

### 8. "The" Social Credit System (61.1 min)

**Ground Truth:**
> The Chinese Social Credit System (SCS) has been discussed a lot in Western media. However, we do not know currently how the system that is supposed to take nationwide effect by 2020 will look like...

**Kairos Generated:**
> A formal event features Toni and other speakers analyzing China's social credit system, addressing misconceptions, societal impacts, systemic biases, data privacy, and regional variations...

---

### 9. Automotive Ethernet PHY bring-up (60.4 min)

**Ground Truth:**
> I would like to share my experience bringing up various Automotive Ethernet Gigabit PHYs on an iMX8 platform. Agenda: - PHY configuration CheckList...

**Kairos Generated:**
> Jean-Louis Tekekara delivers a technical presentation on Automotive Ethernet PHY bring-up, covering configuration, debugging, and implementation, while addressing audience questions...

---

### 10. Microsoft Azure Web Jobs (67.1 min)

**Ground Truth:**
> The new Web Job in the Microsoft Azure Platforms allows you to run background workloads easily and effortlessly as support for your Microsoft Azure Web Sites...

**Kairos Generated:**
> Magnus Martinsson explains Azure Web Jobs, covering setup, deployment, error handling, scalability, and practical use cases, emphasizing simplicity, cost-efficiency, and audience engagement...

---

## Notes

- Videos 1, 3 hit Gemini BatchEmbed 100-request limit (>100 scenes); synopsis may be incomplete.
- Video 8 ("The" Social Credit System) hit Gemini 429 rate limit during LLM synopsis — may have incomplete synopsis.
- All videos processed with `execution_mode="sequential"` and `LOW_MEM_MODE=TRUE`.
- ROUGE-L and BLEU scores are low because Kairos produces narrative scene descriptions while TIB ground truth consists of academic abstracts — fundamentally different writing styles.
- BERTScore captures semantic similarity better (0.59 F1), reflecting that Kairos correctly identifies video topics.
