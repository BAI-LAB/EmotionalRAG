import argparse
import json

from personality_tests import personality_assessment

from utils import logger_main as logger

parser = argparse.ArgumentParser()
parser.add_argument(
    "--questionnaire",
    default="16Personalities",
    choices=["16Personalities", "BFI"],
    help="Questionnaire",
)
parser.add_argument(
    "--dataset",
    default="charactereval",
    choices=["charactereval", "characterllm", "incharacter"],
    help="Dataset",
)
parser.add_argument(
    "--agent_llm",
    default="qwen",
    choices=["chatglm", "qwen", "gpt-3.5", "gpt-4"],
    help="Agent LLM",
)
parser.add_argument(
    "--retrieval_method",
    default="C-A",
    choices=["OriginalRAG", "C-M", "C-A", "S-C", "S-S"],
    help="Retrieval Method",
)
args = parser.parse_args()

questionnaire = args.questionnaire
dataset = args.dataset
agent_llm = args.agent_llm
retrieval_method = args.retrieval_method

interview_folder_path = f"result/{dataset}/{agent_llm}/{retrieval_method}"

eval_method = "interview_assess_batch_anonymous"
eval_llm = "gpt-3.5"
repeat_times = 1

with open("data/charactereval/characters_labels.json", "r") as f:
    character_labels = json.load(f)
characters = list(character_labels["pdb"].keys())

logger.info("Start evaluation ...")
results = {}
cnt = 0
for character in characters:
    cnt += 1
    print(cnt)

    result = personality_assessment(
        character,
        dataset,
        agent_llm,
        questionnaire,
        eval_method,
        eval_llm,
        repeat_times=repeat_times,
        retrieval_method=args.retrieval_method,
        dataset=dataset,
    )
    results[(character, dataset)] = result

logger.info(
    "Questionnaire: {}, Eval Method: {}, Repeat Times: {}, Agent LLM: {}, Eval LLM: {}".format(
        questionnaire, eval_method, repeat_times, agent_llm, eval_llm
    )
)

from utils import avg

personality_consistency = {}

for analysis_key in result["analysis"].keys():
    analysis_values = [v["analysis"][analysis_key] for v in results.values()]
    analysis_value = avg(analysis_values)

    logger.info("Analyzing {}: {:.4f}".format(analysis_key, analysis_value))
    personality_consistency[analysis_key] = analysis_value

preds = {
    rpa: {dim: result["dims"][dim]["all_scores"] for dim in result["dims"]}
    for rpa, result in results.items()
}

if questionnaire in ["BFI", "16Personalities"]:
    label_settings = ["pdb"]
    labels_pdb = {
        rpa: {
            dim: character_labels["pdb"][rpa[0]][questionnaire][dim]
            for dim in result["dims"]
        }
        for rpa, result in results.items()
    }
else:
    label_settings = ["annotation"]
    labels_pdb = {
        rpa: {
            dim: character_labels["annotation"][rpa[0]][questionnaire][dim]
            for dim in result["dims"]
        }
        for rpa, result in results.items()
    }

for label_setting in label_settings:
    labels = {
        rpa: {
            dim: character_labels[label_setting][rpa[0]][questionnaire][dim]
            for dim in result["dims"]
        }
        for rpa, result in results.items()
    }  # e.g. { "score": 65.88130032806441, "type": "H"}

    from personality_tests import calculate_measured_alignment

    measured_alignment = calculate_measured_alignment(
        preds, labels, questionnaire, labels_pdb=labels_pdb
    )

    single_acc = measured_alignment["all"]["single_acc"]["all"]
    single_mse = measured_alignment["all"]["single_mse"]["all"]
    single_mae = measured_alignment["all"]["single_mae"]["all"]
    full_acc = measured_alignment["all"]["full_acc"]

    logger.info("Single Acc, Full Acc, Single MSE, Single MAE")
    logger.info(
        "Alignment {}: {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(
            label_setting.upper()[:3], single_acc, full_acc, single_mse, single_mae
        )
    )
