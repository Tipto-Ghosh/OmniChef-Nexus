# OmniChef-Nexus
OmniChef-VL-RAG is a multimodal retrieval-augmented generation (RAG) system designed for intelligent recipe understanding and interaction. The system supports both visual and textual inputs, enabling users to discover, explore, and discuss recipes through a unified AI-driven interface.




* Plan is to build a multi-modal RAG pipeline for Food Recipes.

* we want to use this dataset
    - Dataset: https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions

* What we actually want to build:

- So nvidia recently launched their nemotron embedding series with re-ranker model. We want to build a multi-modal food recipe rag application which can do the following:

   - Suggest recipes based on user queries
      - queries can be text only, image only or combination of this two.


* Overall pipeline:
  Dataset -> pdf+image doc -> embedding -> store -> retrive -> re-rank -> generation based on user query