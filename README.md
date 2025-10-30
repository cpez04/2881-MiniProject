# 2881-MiniProject




steps for reproduction:
1. run reproduce/wikiqa.ipynb to extract the 230 longest wiki qa questions
2. run reproduce/wikicontent.ipynb to get 1165 of the most recent wiki articles after nov 7, 2023 (doing this cuz the data in the repo is terribly formatted)
3. run reproduce/chunk.ipynb - chunks the articles into 256 token w stride of 128
4. run reproduce/bm25.ipynb - simulates rag retrieval, getting the top k=1 chunks per querym resulting in retrieved_contexts.jsonl