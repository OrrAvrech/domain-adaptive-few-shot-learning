import json
import spacy
from pathlib import Path
import pandas as pd


def read_file(filepath: Path) -> dict:
    with open(filepath, "r") as fp:
        data = json.load(fp)
    return data


def similar_words(word1: str, word2: str, model: spacy.Language, thresh: float) -> bool:
    # add edit distance to capture non-words with typos or missing spaces
    score = 0.0
    word1 = model(word1)
    word2 = model(word2)
    try:
        score = word1.similarity(word2)
    except KeyError:
        pass

    if score >= thresh:
        return True
    else:
        return False


def process_annotations(df: pd.DataFrame, sim_thresh: float):
    df = df[df["skip"] == False]
    nlp = spacy.load("en_core_web_sm")
    df["new_class"] = df.apply(
        lambda x: x["folder"] if similar_words(x["folder"], x["class"], nlp, sim_thresh) else None,
        axis=1)
    df = df.dropna()
    return df


def main():
    results_dir = Path("/Users/orrav/Documents/Data/domain-adaptive-few-shot-learning/results")
    csv_path = Path("/Users/orrav/Documents/Data/domain-adaptive-few-shot-learning/annotations.csv")
    result_files = [p for p in results_dir.rglob("*") if not p.is_dir()]
    sim_thresh = 0.5

    data = []
    for res_path in result_files:
        print(f"results file {res_path}")
        content = read_file(res_path)
        annotator_id = content["completed_by"]["id"]
        img_url = str(content["task"]["data"]["image"])
        img_name = img_url.split("/")[-1]
        img_folder = img_url.split("/")[-2]
        res = content["result"]
        clarity, abstraction = 0, 0
        cls = ""
        skip = False
        if len(res) == 0:
            skip = True
        else:
            try:
                clarity = res[0]["value"]["rating"]
                abstraction = res[1]["value"]["rating"]
                cls = res[2]["value"]["text"][0]
            except (IndexError, KeyError):
                print(f"skip results file {res_path.name}")

        row = {"img_name": img_name, "folder": img_folder, "class": cls,
               "clarity": clarity, "abstraction": abstraction,
               "annotator": annotator_id, "skip": skip}
        data.append(row)

    df = pd.DataFrame(data)
    # df.to_csv(csv_path)
    filtered_df = process_annotations(df, sim_thresh)


if __name__ == '__main__':
    main()
