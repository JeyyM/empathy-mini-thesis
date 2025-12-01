# Text Summarization System - Complete Inventory

## 📋 Overview
This system evaluates the quality of participant-generated summaries of their ChatGPT conversations. Participants had debates with ChatGPT on controversial topics, then wrote summaries describing the main points and outcomes.

---

## 🎯 Purpose

**Research Question:** Can participants accurately summarize the content and tone of their empathetic conversations?

**Process:**
1. Participants engage in ChatGPT debates on controversial topics
2. Participants write their own text summaries of the conversation
3. The grading system evaluates summary quality against the original conversation

---

## 📁 Folder Structure

```
text summarization 2/
├── files/
│   ├── neutral/          # 5 participants (Neutral group)
│   ├── opposing/         # 5 participants (Opposing group)
│   └── similar/          # 5 participants (Similar group)
├── main.py               # Core grading algorithm
├── run_all_grading.py    # Batch processing script
├── grading_results.csv   # Output: All scores and metrics
├── exchange.txt          # Example: Original conversation
├── summary.txt           # Example: Participant summary
├── summary2.txt          # Example 2
├── summary3.txt          # Example 3
└── requirements.txt      # Dependencies
```

---

## 📂 Input Data Files

### **Per Participant: 2 Text Files**

#### **1. [Name]Chat.txt** - Original ChatGPT Conversation
**Example:** `MiguelBorromeoChat.txt`

**Contents:**
- Complete transcript of ChatGPT conversation
- Includes both user and ChatGPT messages
- Format: "You said:" and "ChatGPT said:" alternating
- Contains metadata: "Skip to content", timestamps, etc.
- Length: ~1,000-3,000 words

**Purpose:** The **ground truth** - what actually happened in the conversation

---

#### **2. [Name]Summary.txt** - Participant's Written Summary
**Example:** `MiguelBorromeoSummary.txt`

**Contents:**
- Participant's own description of the conversation
- Written in their own words (paraphrasing expected)
- No fixed format or length requirement
- Length: ~30-180 words (highly variable)

**Purpose:** The **evaluation target** - how well did they remember/summarize?

**Example Summary:**
```
"The main points were that the death penalty is a major human rights 
issue. Other alternative methods have been explored so we don't resort 
to killing people because that is the ultimate end all be all like you 
can't reverse that and in cases of wrongful convictions, that is 
problematic. And there were also other ways to punish usually for 
non-violent crimes, there are other ways to deter such as fines and 
sentences."
```

---

## 🔧 Core Grading System

### **`main.py` - Content Recall Grader**

**Class:** `ContentRecallGrader`

#### **What it does:**
Evaluates how well a summary captures the original conversation's meaning and key points.

#### **Philosophy:**
- **NOT checking for exact words** - paraphrasing is expected and encouraged
- **NOT checking grammar/spelling** - focus is on content recall
- **NOT checking writing quality** - focus is on meaning preservation
- **IS checking:** Did they understand and remember what was discussed?

---

### **Three Core Metrics**

#### **1. Semantic Similarity** (25% weight)
**Measures:** Overall word/phrase overlap using TF-IDF cosine similarity

**How it works:**
- Uses sklearn's TfidfVectorizer with 1-3 word n-grams
- Compares semantic vector space between original and summary
- Fallback: Simple word overlap if sklearn unavailable

**Scoring:**
- 1.0 = Perfect semantic match
- 0.0 = No semantic overlap

**Human Adjustment:** Multiplied by 2.4× to account for natural paraphrasing
- Humans naturally rephrase instead of copying verbatim
- Boost ensures paraphrasing isn't penalized

**Output:** `Semantic_Similarity` (0-1 scale)

---

#### **2. Topic Coverage** (65% weight) - **MOST IMPORTANT**
**Measures:** Are the main topics/themes from the conversation covered?

**How it works:**
1. Extract important words from original (frequent meaningful words)
2. Remove stop words (the, and, is, etc.)
3. Identify top 15 most important topics
4. Check how many appear in summary (or related terms)

**Scoring:**
- 1.0 = All main topics covered
- 0.0 = No topics covered

**Human Adjustment:** Multiplied by 1.6× for natural theme coverage

**Output:** `Topic_Coverage` (0-1 scale)

---

#### **3. Factual Accuracy** (10% weight)
**Measures:** Are there contradictions or false claims?

**How it works:**
- Starts at 1.0 (perfect)
- Checks for contradictory sentiment pairs:
  - Positive vs negative tone
  - Agreement vs disagreement claims
  - Success vs failure descriptions
- Deducts 0.2 for each clear contradiction

**Scoring:**
- 1.0 = No contradictions
- 0.0 = Major factual errors

**Output:** `Factual_Accuracy` (0-1 scale)

---

### **Overall Score Calculation**

**Formula:**
```
Overall = (Semantic_Similarity × 0.25) + (Topic_Coverage × 0.65) + (Factual_Accuracy × 0.10)
Percentage = Overall × 100
```

**Letter Grades:**
- **A:** ≥75% - Excellent content recall
- **B:** 60-74% - Good content recall
- **C:** 45-59% - Adequate content recall
- **D:** 30-44% - Poor content recall
- **F:** <30% - Very poor content recall

**Note:** Grading scale is calibrated for **human-written summaries** where paraphrasing is natural and expected.

---

## 🚀 Execution Scripts

### **`run_all_grading.py` - Batch Processing**

**What it does:**
1. Scans all three group folders (neutral, opposing, similar)
2. For each participant:
   - Loads [Name]Chat.txt (original)
   - Loads [Name]Summary.txt (participant summary)
   - Runs ContentRecallGrader
   - Stores results
3. Calculates group averages
4. Exports to CSV

**Output Format:**
```
GROUP: NEUTRAL
MiguelBorromeo:
  Score: 58.5% (Grade: C)
  - Semantic Similarity: 0.361
  - Topic Coverage: 0.608
  - Factual Accuracy: 1.000
```

**Final Output:** `grading_results.csv`

---

## 📊 Output Data: `grading_results.csv`

### **Columns (9 total):**

1. **Group** - Participant group (neutral, opposing, similar)
2. **Name** - Participant name
3. **Overall_Percentage** - Final score (0-100%)
4. **Letter_Grade** - A, B, C, D, or F
5. **Semantic_Similarity** - Word/phrase overlap score (0-1)
6. **Topic_Coverage** - Theme coverage score (0-1)
7. **Factual_Accuracy** - Contradiction check score (0-1)
8. **Original_Words** - Word count of original conversation
9. **Summary_Words** - Word count of participant summary
10. **Compression_Ratio** - Summary words / Original words

---

## 📈 Results Summary

### **Participant Data (15 total):**

#### **Neutral Group (5 participants):**
- MiguelBorromeo: 58.5% (C)
- MiguelNg: 35.8% (D)
- RandellFabico: 26.2% (F)
- RusselGalan: 42.9% (D)
- RyanSo: 65.4% (B)

**Group Average:** 45.8%

#### **Opposing Group (5 participants):**
- AaronDionisio: 34.2% (D)
- ArianPates: 57.8% (C)
- EthanPlaza: 75.2% (A)
- MarwahMuti: 85.4% (A)
- SamuelLim: 38.2% (D)

**Group Average:** 58.2%

#### **Similar Group (5 participants):**
- AndreMarco: 26.5% (F)
- EthanOng: 45.9% (C)
- KeithziCantona: 27.3% (F)
- MaggieOng: 55.0% (C)
- SeanTe: 45.5% (C)

**Group Average:** 40.0%

---

## 🔍 Key Patterns in Data

### **Summary Length Patterns:**
- **Shortest:** 31 words (AndreMarco)
- **Longest:** 180 words (RyanSo)
- **Average:** ~90 words
- **Compression ratio:** 2-8% of original (most summaries are 3-5%)

### **Score Distribution:**
- **A grades:** 2/15 (13%) - Both in Opposing group
- **B grades:** 1/15 (7%)
- **C grades:** 5/15 (33%)
- **D grades:** 4/15 (27%)
- **F grades:** 3/15 (20%)

### **Group Performance:**
1. **Opposing:** 58.2% average (Best)
2. **Neutral:** 45.8% average
3. **Similar:** 40.0% average (Worst)

**Insight:** Opposing group had better content recall than Similar group

---

## 🔬 Technical Details

### **Dependencies (requirements.txt):**
```
scikit-learn  # For TF-IDF vectorization and cosine similarity
numpy         # For numerical operations
```

**Optional:** System works with fallback if sklearn unavailable (uses simple word overlap)

### **Text Cleaning:**
Removes ChatGPT conversation artifacts:
- "Skip to content"
- "You said:" / "ChatGPT said:"
- "Report conversation"
- Timestamps and metadata
- Extra whitespace

### **Processing Time:**
- Per summary: <1 second
- All 15 summaries: <5 seconds

---

## 📝 Example Grading Breakdown

**Participant:** MiguelBorromeo (Neutral group)

**Original Conversation:** 1,185 words (debate about death penalty)

**Summary:** 73 words
```
"The main points were that the death penalty is a major human rights 
issue. Other alternative methods have been explored so we don't resort 
to killing people because that is the ultimate end all be all like you 
can't reverse that and in cases of wrongful convictions, that is 
problematic. And there were also other ways to punish usually for 
non-violent crimes, there are other ways to deter such as fines and 
sentences."
```

**Scores:**
- **Semantic Similarity:** 0.361 (after 2.4× boost)
  - Raw overlap is moderate, boosted for paraphrasing
- **Topic Coverage:** 0.608 (after 1.6× boost)
  - Captures main themes: death penalty, human rights, irreversibility, wrongful convictions, alternatives
- **Factual Accuracy:** 1.0
  - No contradictions detected

**Overall Calculation:**
```
(0.361 × 0.25) + (0.608 × 0.65) + (1.0 × 0.10) = 0.585
0.585 × 100 = 58.5%
Grade: C (Adequate content recall)
```

**Compression:** 73/1185 = 6.2% (summary is 6% of original length)

---

## 🎯 Use in Thesis

### **Integration with Emotion Data:**
Summary quality scores are **correlated with emotion/cognition features** to test:
- Does empathetic emotion during conversation predict better comprehension?
- Does cognitive load affect summary quality?
- Are certain groups better at recalling content?

### **Key Variables for Analysis:**
From `grading_results.csv`:
1. `Overall_Percentage` - Overall summary quality (primary outcome)
2. `Semantic_Similarity` - Word-level recall
3. `Topic_Coverage` - Theme-level recall
4. `Factual_Accuracy` - Accuracy of recall
5. `Compression_Ratio` - Summary efficiency

These are merged with:
- `facial_summary_merged.csv` (209 facial features)
- `voice_summary_merged.csv` (375 voice features)
- `fusion_summary_merged.csv` (273 fusion features)

**Research Question:** Can we predict summary quality from emotional/vocal patterns during conversation?

---

## 📚 File Inventory Summary

### **Input Files:**
- **30 text files** (15 participants × 2 files each)
  - 15 Chat files (original conversations)
  - 15 Summary files (participant summaries)
- **3 example files** (exchange.txt, summary.txt, summary2.txt, summary3.txt)

### **Processing Scripts:**
- **2 Python scripts**
  - `main.py` (grading algorithm)
  - `run_all_grading.py` (batch processor)

### **Output Files:**
- **1 CSV file**
  - `grading_results.csv` (15 rows × 10 columns)

### **Configuration:**
- **1 requirements file**
  - `requirements.txt` (dependencies)

### **Total Files in System:** ~35 files

---

## 💡 Key Insights

### **Grading Philosophy:**
- **Human-centric:** Designed for natural human summarization
- **Meaning over form:** Paraphrasing encouraged, exact words not required
- **Content recall focus:** Did they understand the conversation?
- **Balanced metrics:** Combines semantic, thematic, and factual accuracy

### **Strengths:**
- ✅ Robust to paraphrasing (2.4× boost on semantic similarity)
- ✅ Emphasizes topic coverage over verbatim recall (65% weight)
- ✅ Automatic batch processing
- ✅ Clear, interpretable scores

### **Limitations:**
- ⚠️ Assumes TF-IDF captures semantic meaning (may miss nuance)
- ⚠️ Topic coverage relies on word frequency (may miss subtle themes)
- ⚠️ Small sample size (n=15, 5 per group)
- ⚠️ No manual validation of scores

---

## 🔄 Workflow

```
Participant ChatGPT Conversations
    ↓
[Name]Chat.txt files (ground truth)
    ↓
Participants write summaries
    ↓
[Name]Summary.txt files (evaluation target)
    ↓
┌─────────────────────────────────────┐
│ run_all_grading.py                  │
│                                     │
│  For each participant:              │
│    1. Load Chat + Summary           │
│    2. main.py: ContentRecallGrader  │
│       - Clean text                  │
│       - Semantic similarity         │
│       - Topic coverage              │
│       - Factual accuracy            │
│    3. Calculate weighted score      │
│    4. Assign letter grade           │
└─────────────────────────────────────┘
    ↓
grading_results.csv
(15 participants × 10 metrics)
    ↓
Merged with emotion/voice data
    ↓
Statistical analysis & prediction models
```

---

**End of Text Summarization System Inventory**
