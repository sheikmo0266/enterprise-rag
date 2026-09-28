# Multi-Document Enterprise RAG

An intelligent document question-answering application built using Retrieval-Augmented Generation (RAG). The system processes multiple PDF documents, retrieves relevant information using semantic search, and generates context-aware responses using a Large Language Model (LLM).

## Overview

This project demonstrates the implementation of a RAG pipeline for document-based question answering. Instead of relying only on the knowledge stored within an LLM, the system retrieves relevant information from uploaded documents and provides it as context to the LLM before generating a response.

## Objective

The objective of this project is to develop an intelligent multi-document question-answering system using Retrieval-Augmented Generation (RAG). The system aims to enable users to efficiently retrieve relevant information from multiple PDF documents and generate accurate, context-aware responses using an LLM.

## Key Features

- Multi-document PDF processing
- Document text extraction and preprocessing
- Text chunking for efficient retrieval
- Semantic text embeddings
- FAISS-based vector similarity search
- Retrieval of relevant document context
- Context-aware LLM response generation
- Document-based question answering

## RAG Workflow

```text
PDF Documents
      ↓
Document Processing
      ↓
Text Chunking
      ↓
Text Embeddings
      ↓
FAISS Vector Index
      ↓
Semantic Search
      ↓
Relevant Context
      ↓
LLM
      ↓
Context-Aware Answer