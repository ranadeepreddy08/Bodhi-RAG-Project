# 📚 Bodhi AI Tutor

### AI Textbook Tutor in Your Mother Tongue

Bodhi is a multilingual, textbook-grounded AI tutor designed for Indian students who may study from English-medium textbooks but understand concepts better in their mother tongue.

Instead of giving generic chatbot answers, Bodhi grounds its explanations in the student's uploaded textbook and teaches concepts in a simple, student-friendly regional language.

---

## 🎯 Problem

Many Indian students study from English-medium textbooks while thinking and reasoning in their mother tongue.

The problem is not always a lack of ability. Often, the explanation simply does not arrive in the language in which the student understands the concept best.

Generic AI chatbots can make this worse by:

- Giving polished but generic answers
- Ignoring the student's actual textbook
- Hallucinating information outside the textbook
- Assuming that the student understood the explanation
- Providing little personalized practice

Bodhi is designed to address these problems through textbook-grounded RAG, regional-language teaching, and a teach-back learning loop.

---

## 💡 Solution

Bodhi follows this pipeline:

```text
Student uploads textbook
          ↓
PDF / textbook extraction
          ↓
Text cleaning and chunking
          ↓
BGE embeddings
          ↓
Chroma vector database
          ↓
Relevant textbook retrieval
          ↓
Grounded LLM response
          ↓
Regional-language explanation
          ↓
Teach-back evaluation
          ↓
Adaptive practice