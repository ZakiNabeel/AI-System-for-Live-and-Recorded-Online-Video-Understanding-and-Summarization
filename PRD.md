# Project Title

Design and Implementation of an AI System for Live and Recorded Online Video
Understanding and Summarization

# Project Objective

Develop an intelligent software system capable of analyzing:

1. Live online video streams (e.g., YouTube Live, webinars, lectures)
2. Pre-recorded online videos (e.g., existing YouTube videos)

The system shall process spoken and visual content, extract important information, and generate
summaries, insights, and event detections automatically.

# Problem Statement

Online videos contain large amounts of information that are difficult to analyze manually. The
objective of this project is to design an AI-based system that can process both real-time live
streams and existing recorded online videos, understand their content, and generate meaningful
summaries and analysis.

# Required Tasks

## Module 1: Live Video Analysis

Students shall develop a system to:

```
 Capture and process live video streams
 Convert speech to text in near real time
 Analyze visual content from extracted frames
 Generate rolling live summaries
```
## Module 2: Recorded Video Analysis (YouTube Video Analyzer)

Students shall also develop a pipeline that accepts a YouTube video URL and performs:

```
 Video retrieval from online source
 Transcript extraction (or speech transcription if transcript unavailable)
 Timestamp extraction
 Key-frame or screenshot extraction
 Visual content analysis
 Topic/event detection
 Automatic video summarization
```

Output should include:

```
 Full summary
 Time-stamped key points
 Important detected events
 Optional question answering about the video
```
# Minimum Functional Requirements

The system must support:

✔ Live video analysis

✔ Existing YouTube video analysis

✔ Speech understanding

✔ Visual understanding

✔ Multimodal summarization

✔ Time-stamped insights

# Suggested Pipeline

**Input:
Live Stream OR YouTube URL**

**↓**

**Audio Extraction
Speech-to-Text**

**↓**

**Frame / Key Screenshot Extraction
Visual Analysis**

**↓**

**LLM / Multimodal Reasoning**

**↓**

**Outputs:**

**- Transcript
- Summary
- Event Detection**


**- Time-stamped Insights**

# Deliverables

Students must submit:

1. Source Code
2. Architecture Design
3. Demo using:
    o One live stream
    o One existing YouTube video
4. Performance Evaluation Report
5. Final Project Report

# Bonus Extensions (Optional)

Extra credit for:

```
 Strategy extraction from tutorial videos
 Domain-specific analysis (medical, trading, law, education)
 Generate rules or workflows from analyzed videos
 Convert extracted strategy into pseudocode or executable logic
```

