# Career-Driven Course Recommender

A content-based course recommendation system that takes a computer
science student's target career and returns a ranked list of ten
Coursera courses relevant to that career. Built as a Master's
capstone project at the University of Sydney, this repository
contains the source code, datasets, generated profiles, and
evaluation results for the thesis *Personalising Course
Recommendations from a Career Goal Using Content-Based Filtering
and Large Language Models*.

The system maps each career to an occupation in the O*NET database,
builds a query from that occupation, scores Coursera courses using
TF-IDF and cosine similarity, and re-ranks the top candidates by
course rating. Three query-construction methods are compared: P1
uses the O*NET occupational description directly, P2 uses a large
language model profile grounded in the O*NET description, and P3
uses a large language model profile generated from the career title
alone. The three pipelines were evaluated on 130 students across 17
careers, using five metrics (HR@10, P@10, P@5, MRR, nDCG@10), and
both language model pipelines scored higher than the direct
pipeline on every metric.

## Repository structure

    datasets/
        cleaned/      Cleaned datasets used by the pipelines
        raw/          Original Kaggle and O*NET sources
    src/
        pre_process/      Data cleaning and preprocessing
        recommendations/  The three pipelines (P1, P2, P3)
        evaluation/       Metric computation and evaluation
    results/
        recommendations/  Generated recommendations per pipeline
        evaluation/       Per-student, per-career, and overall
                          metric results

## Requirements

Python 3.10+, pandas, scikit-learn, openpyxl (to read the O*NET
spreadsheet), and anthropic (for the LLM-based pipelines P2 and P3).
Install with:

    pip install pandas scikit-learn openpyxl anthropic

## How to run

First preprocess the student dataset, which produces the working file
used by the pipelines and the evaluation script:

    python src/pre_process/preprocess_students_dataset.py

Then run any of the three pipelines, which write their
recommendations to `results/recommendations/tfidf/`:

    python src/recommendations/tfidf/p1_onet.py
    python src/recommendations/tfidf/p2_llm_onet.py
    python src/recommendations/tfidf/p3_llm_title.py

Pipelines P2 and P3 require an Anthropic API key, set as an
environment variable:

    export ANTHROPIC_API_KEY=your_key_here

Finally, run the evaluation, which computes the five metrics per
student and per career and writes results to `results/evaluation/`:

    python src/evaluation/eval.py

## Datasets

The system uses three datasets. The CS Students dataset is a
publicly-available student career-aspiration dataset from Kaggle
(Nusrat, 2024), with 180 students of which 130 are retained after
excluding careers without a sound O*NET match. The Coursera Courses
dataset is a publicly-available Coursera dataset from Kaggle
(Kapoor, 2021), with 3,424 courses, including name, description,
rating, and Skills tags. The O*NET Occupations dataset contains
1,016 occupational descriptions from the U.S. Department of Labor's
O*NET database.

## Results summary

On the 130-student dataset, both language model pipelines scored
higher than the direct pipeline on every metric.

    Pipeline   HR@10   P@10   P@5    MRR    nDCG@10
    P1          87.7   42.9   47.4   71.8    70.6
    P2         100.0   77.1   76.3   84.2    88.4
    P3          98.5   75.5   82.0   84.6    89.7

Full per-career results are reported in the thesis (Appendix E).

## Notes

This is a Master's capstone project. The system is content-based and
targets the cold-start setting: the only input is the student's
career goal, and every student who shares a career receives the same
recommendations. The Anthropic API key has been removed from the
source — set it via the environment variable above to run P2 and P3.

## Author

Ahmed Fadhl Allah K Albulushi
Master of Engineering, University of Sydney
Supervisor: Dr. Huaming Chen
