# Emotional RAG: Enhancing Role-Playing Agents through Emotional Retrieval

This repository contains the code for the paper "Emotional RAG: Enhancing Role-Playing Agents through Emotional Retrieval," which has been accepted by ICKG2024. The project aims to enhance the capabilities of role-playing agents through emotional retrieval.

## Environment Requirements

- Python Version: 3.12
- Other Library Versions: See `requirements.txt`

## Installation Steps

### STEP 1: Prepare Models and Configuration

1. **Download Base Models**:
   - [Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)
   - [BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5)

2. **Configure Model Paths and GPT Keys**:
   - Configure local model paths and GPT keys in `utils/config.py`.
   - Configure GPT keys for evaluation in `evaluation/config.json`.

### STEP 2: Prepare Datasets

1. **Download CharacterEval Dataset**:
   ```bash
   wget https://github.com/morecry/CharacterEval/raw/refs/heads/main/data/test_data.jsonl
   wget https://raw.githubusercontent.com/morecry/CharacterEval/refs/heads/main/data/character_profiles.json
   ```
   Place the files from the [CharacterEval dataset](https://github.com/morecry/CharacterEval/tree/main/data) into the `data/CharacterEval` directory.

2. **Generate query_bank**:
   ```bash
   python data/generate_query_bank.py
   ```

3. **Generate the All Memory Bank for All Characters**:
   ```bash
   python data/charactereval/generate_qa_memory_bank.py
   ```

### STEP 3: Generate Psychology Questionnaire Answers and Evaluate

1. **Generate Psychology Questionnaire Answers**:
   ```bash
   python get_response.py
   ```

2. **Conduct Evaluation**:
   ```bash
   python evaluation/run_experiments.py
   ```
   The evaluation plan uses the personality assessment from [InCharacter](https://github.com/Neph0s/InCharacter).
## Citation

If you use this project in your research, please cite the following paper:

```
@misc{huang2024emotionalragenhancingroleplaying,
      title={Emotional RAG: Enhancing Role-Playing Agents through Emotional Retrieval}, 
      author={Le Huang and Hengzhi Lan and Zijun Sun and Chuan Shi and Ting Bai},
      year={2024},
      eprint={2410.23041},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2410.23041},  
}
```
