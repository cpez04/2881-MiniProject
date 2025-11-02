# 2881-MiniProject

## Project Overview

This project contains multiple components for studying RAG (Retrieval-Augmented Generation) systems and data leakage in the context of the paper ``Follow My Instruction and Spill the Beans: Scalable Data Extraction from Retrieval-Augmented Generation Systems."

The project is organized into several subfolders.

## Quick Setup

1. Clone this repository
2. Install dependencies for each component as needed (see individual README files)
3. Follow the reproduction steps below for the main pipeline

## Reproduction Steps

1. Run `reproduce/wikiqa.ipynb` to extract the 230 longest wiki QA questions
2. Run `reproduce/wikicontent.ipynb` to get 1165 of the most recent wiki articles after Nov 7, 2023
3. Run `reproduce/chunk.ipynb` - chunks the articles into 256 tokens with stride of 128
4. Run `reproduce/bm25.ipynb` - simulates RAG retrieval, getting the top k=1 chunks per query, resulting in `retrieved_contexts.jsonl`
5. Run `reproduce/queries.ipynb` to prepare the queries into a JSONL format.
6. Run `reproduce/together.ipynb` to run the queries against a given model in Together AI. Note: you will need a Together AI API key for this step.

## Project Structure

For detailed setup instructions and explanations of each component, please refer to the individual README files in each subfolder:

- **`extension1/`** - Fine-tuning experiments and synthetic dataset creation
- **`extension2/`** - Data leakage detection and firewall evaluation
- **`reproduce/`** - Contains all the reproduction scripts previously mentioned
- **`ablation/`** - Ablation study on the repetition penalty.
- **`data/`** - Generated datasets and intermediate files from reproduction.
- **`results/`** - Model evaluation results from reproduction.