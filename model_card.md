# DocuBot Model Card

This model card is a short reflection on your DocuBot system. Fill it out after you have implemented retrieval and experimented with all three modes:

1. Naive LLM over full docs
2. Retrieval only
3. RAG (retrieval plus LLM)

Use clear, honest descriptions. It is fine if your system is imperfect.

---

## 1. System Overview

**What is DocuBot trying to do?**

> _DocuBot is attempting to act as a document assistant that retrieves document snippets and answers questions based on the information it has available. It retrieves documents, scores them, and returns answers to a developer's questions based on evidence, which allows it to refuse to answer questions that are irrelevant or have too little evidence to formulate an answer._

**What inputs does DocuBot take?**  
For example: user question, docs in folder, environment variables.

> _DocuBot takes in user queries, text documents formatted in markdown, and a Gemini API key_

**What outputs does DocuBot produce?**

> _DocuBot outputs succinct answers to questions that can be answered based on existing documentation._

---

## 2. Retrieval Design

**How does your retrieval system work?**  
Describe your choices for indexing and scoring.

- How do you turn documents into an index?
- How do you score relevance for a query?
- How do you choose top snippets?

> _The index is created by chunking documents based on their markdown headings into tuples (filename, section_text) and associating each word with the file in which it appears. Relevance is scored by simply incrementing a score for each word in a query based on how many times it appears in the documentation. Top snippets are determined by choosing the files in which each word of the query appears and collects each section_text into a set of candidates. These are scored and are filtered so that chunks that score less than 2 are removed, indicating weak relevance. Chunks are sorted by score in descending order._

**What tradeoffs did you make?**  
For example: speed vs precision, simplicity vs accuracy.

> _The retrieval uses a simple algorithm, so it may not be very accurate. It is also dependent on the way that a markdown file is formatted. Text files will likely break the model as it cannot properly chunk the information, resulting in less accuracy._

---

## 3. Use of the LLM (Gemini)

**When does DocuBot call the LLM and when does it not?**  
Briefly describe how each mode behaves.

- Naive LLM mode:
- Retrieval only mode:
- RAG mode:

> DocuBot calls on the LLM in Naive LLM mode and RAG mode.
> Naive LLM mode: The LLM is searching the internet as if the query is a direct question to the LLM itself. It can fabricate answers based on what it researches.
> Retrieval only mode: The LLM is not involved. It is based on an algorithm which returns information based on the score of each word in a query, and its match to the files provided.
> RAG mode: The LLM works with the information returned by the retrieval method, and formulates its answer based on that. This results in succinct answers.

**What instructions do you give the LLM to keep it grounded?**  
Summarize the rules from your prompt. For example: only use snippets, say "I do not know" when needed, cite files.

> _The DocuBot needs to refuse to answer if it has too low an amount of evidence to confidently answer a question. It should not make a confident guess._

---

## 4. Experiments and Comparisons

Run the **same set of queries** in all three modes. Fill in the table with short notes.

You can reuse or adapt the queries from `dataset.py`.

| Query                                               | Naive LLM: helpful or harmful? | Retrieval only: helpful or harmful? | RAG: helpful or harmful? | Notes |
| --------------------------------------------------- | ------------------------------ | ----------------------------------- | ------------------------ | ----- |
| Example: Where is the auth token generated?         |                                |                                     |                          |       |
| Example: How do I connect to the database?          |                                |                                     |                          |       |
| Example: Which endpoint lists all users?            |                                |                                     |                          |       |
| Example: How does a client refresh an access token? |                                |                                     |                          |       |

**What patterns did you notice?**

- When does naive LLM look impressive but untrustworthy?
- When is retrieval only clearly better?
- When is RAG clearly better than both?

> _Using Naive LLM mode results in answers that are just the LLM searching the internet. It's as if you are talking directly with a chatbot, so it cannot summarize based on your specific use-cases which are based on your specific architecture. It does not see the documentation to refine its answers. Retrieval only is better than Naive LLM when the prompts are worded in a way where words in the query exist in the documentation. This will at least return referenced documentation and the exact passages where the evidence likely exists (based on mathematical determination). RAG is better than both since it can take in the retrieved documentation, and make inferences based on that, providing simple and clean answers._

---

## 5. Failure Cases and Guardrails

**Describe at least two concrete failure cases you observed.**  
For each one, say:

- What was the question?
- What did the system do?
- What should have happened instead?

> _For Retrieval only mode, the question "How do I connect to the database?" results in three documents being referenced that don't answer the question because the scoring system does not check the words "connecting" or "connection", so it missed scoring DATABASE.md entirely. It should have been able to check this similar word and determine that there is relevant information in this document._

> _Because of failure case 1, the retrieval returns three documents that are actually irrelevant to the query, and this results in the RAG method failing because even though there are 3 documents returned by the retrieval, the LLM determines that none of them are relevant and tells the user that it cannot answer the question based on the provided docs. _

**When should DocuBot say “I do not know based on the docs I have”?**  
Give at least two specific situations.

> _1. When being asked questions about completely unrelated topics, like "What are frogs?" since there are no relevant documents that are returned with these words. and 2. when being asked questions about information that is just beyond the scope of available documentation, even if it is related. It should say "I do not know based on the docs I have" when it cannot completely answer a question with just limited information, and it shouldn't provide only half an answer just because it has some of the needed information._

**What guardrails did you implement?**  
Examples: refusal rules, thresholds, limits on snippets, safe defaults.

> _Your answer here._

---

## 6. Limitations and Future Improvements

**Current limitations**  
List at least three limitations of your DocuBot system.

1. _Limitation 1_
2. _Limitation 2_
3. _Limitation 3_

**Future improvements**  
List two or three changes that would most improve reliability or usefulness.

1. _Improvement 1_
2. _Improvement 2_
3. _Improvement 3_

---

## 7. Responsible Use

**Where could this system cause real world harm if used carelessly?**  
Think about wrong answers, missing information, or over trusting the LLM.

> _Your answer here._

**What instructions would you give real developers who want to use DocuBot safely?**  
Write 2 to 4 short bullet points.

- _Guideline 1_
- _Guideline 2_
- _Guideline 3 (optional)_

---
