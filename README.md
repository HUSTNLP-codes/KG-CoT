# KG-CoT Chain-of-Thought Prompting of Large Language Models over Knowledge Graphs for Knowledge-Aware Question Answering
Large language models (LLMs) encounter challenges such as hallucination and factual errors in knowledge-intensive tasks. One the one hand, LLMs sometimes struggle to generate reliable answers based on the black-box parametric knowledge, due to the lack of responsible knowledge. Moreover, fragmented knowledge facts extracted by knowledge retrievers fail to provide explicit and coherent reasoning paths for improving LLM reasoning. To address these challenges, we propose KG-CoT, a novel knowledge-augmented paradigm that leverages a small-scale step-by-step graph reasoning model to reason over knowledge graphs (KGs) and utilizes a reasoning path generation method to generate chains of knowledge with high confidence for large-scale LLMs. KG-CoT provides an feasible and effective way to combine traditional multi-hop KGQA techniques with LLM reasoning.

## Setup

- Setup conda environment
```
conda create -n GRT python=3.8
conda activate GRT
```
- Download data

You can download all the preprocessed data with the [link](https://drive.google.com/file/d/1oJJNajXwf-wvpLVshDttcZ1OJh_TFlwY/view?usp=sharing).

## Dependencies
- pytorch
- transformers
- openai
- numpy


## Training
- For WebQSP, WebQuestions and SimpleQuestions
```
cd matrix
python -m train python -m train --input_dir <PATH/TO/DATASET> --save_dir <PATH/TO/SAVE>
```
- For CompWebQ
```
cd analog
python -m train python -m train --input_dir <PATH/TO/DATASET> --save_dir <PATH/TO/SAVE>
```

## Path Generating
- For WebQSP, WebQuestions and SimpleQuestions
```
cd matrix
python -m get_path --input_dir <PATH/TO/DATASET> --save_dir <PATH/TO/SAVE> --ckpt <PATH/TO/CHECKPOINT>
```
- For ComplexWebQ
```
cd analog
python -m get_path --input_dir <PATH/TO/DATASET> --save_dir <PATH/TO/SAVE> --ckpt <PATH/TO/CHECKPOINT>
```

## Joint Reasoning
- We use OpenAI API to call ChatGPT and GPT-4 for experiments, you can follow the jupyter notebook to conduct joint reasoning with reasoning paths.
```
cd llm
LLM_eval.ipynb
```


