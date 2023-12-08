import json
import spacy
import editdistance
from pathlib import Path
import pandas as pd


def read_file(filepath: Path) -> dict:
    with open(filepath, "r") as fp:
        data = json.load(fp)
    return data


def similar_words(
    word1: str,
    word2: str,
    model: spacy.Language,
    wv_thresh: float,
    editdist_thresh: float,
) -> bool:
    # word2vec and editdistance to capture typos and missing spaces
    retval = False
    score = 0.0
    word1_nlp = model(word1)
    word2_nlp = model(word2)
    try:
        score = word1_nlp.similarity(word2_nlp)
    except KeyError:
        pass

    if score >= wv_thresh:
        retval = True
    else:
        edit_dist = editdistance.eval(word1, word2)
        if edit_dist <= editdist_thresh:
            retval = True
    return retval


def process_annotations(df: pd.DataFrame, wv_thresh: float, editdist_thresh: float):
    df = df[~df["skip"]]
    nlp = spacy.load("en_core_web_sm")
    df["final_class"] = df.apply(
        lambda x: x["folder"]
        if similar_words(x["folder"], x["class"], nlp, wv_thresh, editdist_thresh)
        else None,
        axis=1,
    )
    df = df.dropna()
    df = df[~((df["clarity"] == 1) & (df["abstraction"] <= 2))]
    return df


def main():
    root_dir = Path("/Users/orrav/Documents/Data/domain-adaptive-few-shot-learning")
    results_dir = root_dir / "results"
    annotations_path = root_dir / "annotations.csv"
    filtered_path = root_dir / "filtered_annotations.csv"
    result_files = [p for p in results_dir.rglob("*") if not p.is_dir()]
    wv_thresh = 0.5
    editdist_thresh = 2

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

        row = {
            "img_name": img_name,
            "folder": img_folder,
            "class": cls,
            "clarity": clarity,
            "abstraction": abstraction,
            "annotator": annotator_id,
            "skip": skip,
        }
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(annotations_path)
    filtered_df = process_annotations(df, wv_thresh, editdist_thresh)
    filtered_df.to_csv(filtered_path)


if __name__ == "__main__":
    main()
