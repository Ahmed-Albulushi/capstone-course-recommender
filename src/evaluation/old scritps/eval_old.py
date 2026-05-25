# ============================================================
# Evaluation Script — All Pipelines
# ============================================================
# Ground truth:
#   Coursera specific category labels (last token of Skills tag)
#   Each career is mapped to one or more expected Coursera
#   categories. A recommended course is relevant if its
#   specific category matches the expected one.
#
#   Independence guaranteed: recommendation scripts use
#   Course Name + Description only. Skills tag (including
#   category labels) was never seen during recommendation.
#
# Metrics:
#
#   HR@10 (Hit Rate at 10):
#     Fraction of students who received at least one
#     correct-category course in their top 10.
#     Example: 160/174 students got ≥1 machine-learning
#     course → HR@10 = 0.92
#
#   CHR@10 (Category Hit Rate at 10):
#     Average fraction of top-10 courses per student
#     that belong to the expected category.
#     Example: ML Engineer gets 6/10 machine-learning
#     courses → CHR@10 = 0.6
#
#   P@5 (Precision at 5):
#     Average fraction of top-5 courses per student
#     that belong to the expected category.
#     Example: ML Engineer's top 5 has 4 machine-learning
#     courses → P@5 = 0.8
#
#   MRR (Mean Reciprocal Rank):
#     Average of 1/rank of the first correct-category course.
#     Example: first machine-learning course at rank 2
#     → MRR contribution = 1/2 = 0.5
#
#   nDCG@10 (Normalised Discounted Cumulative Gain at 10):
#     Position-weighted relevance — correct courses ranked
#     higher contribute more to the score.
#     Example: correct courses at ranks 1,2,3 scores higher
#     than correct courses at ranks 8,9,10 even if count
#     is the same. Range: 0 to 1, higher is better.
#
# Pipelines evaluated:
#   P1 — Career + O*NET → TF-IDF
#   P2 — Career + O*NET + SFIA → TF-IDF
#   P3 — Career title → LLM → TF-IDF
#   P4 — Career + O*NET → LLM → TF-IDF
#   P5 — Career + O*NET + SFIA → LLM → TF-IDF
#
# Input:
#   - results/recommendations/tfidf/p1_recommendations.csv
#   - results/recommendations/tfidf/p2_recommendations.csv
#   - results/recommendations/tfidf/p3_recommendations.csv
#   - results/recommendations/tfidf/p4_recommendations.csv
#   - results/recommendations/tfidf/p5_recommendations.csv
#
# Output:
#   - results/evaluation/eval_results.csv      (per-student)
#   - results/evaluation/eval_summary.csv      (per-pipeline)
#   - results/evaluation/eval_career.csv       (per-career)
# ============================================================

import os
import pandas as pd
import numpy as np

# ============================================================
#   Paths
# ============================================================

BASE  = '/Users/soesoe/Documents/Capstone Project/final_capstone-course-recommender'
RECO  = os.path.join(BASE, 'results', 'recommendations', 'tfidf')
OUT   = os.path.join(BASE, 'results', 'evaluation')

# ============================================================
#   Pipeline files
# ============================================================

PIPELINES = {
    'P1 — O*NET only':         'p1_recommendations.csv',
    'P2 — O*NET + SFIA':       'p2_recommendations.csv',
    'P3 — LLM (career only)':  'p3_recommendations.csv',
    'P4 — LLM (O*NET only)':   'p4_recommendations.csv',
    'P5 — LLM (O*NET+SFIA)':   'p5_recommendations.csv',
}

# ============================================================
#   Coursera taxonomy
# ============================================================

KNOWN_DOMAINS = {
    'business', 'computer-science', 'data-science', 'life-sciences',
    'physical-science-and-engineering', 'social-sciences', 'arts-and-humanities',
    'information-technology', 'language-learning', 'personal-development', 'math-and-logic'
}

# ============================================================
#   Career → expected Coursera specific category mapping
# ============================================================

CAREER_TO_CATEGORY = {
    'Machine Learning Engineer':    ['machine-learning'],
    'Machine Learning Researcher':  ['machine-learning'],
    'AI Researcher':                ['machine-learning'],
    'Computer Vision Engineer':     ['machine-learning'],
    'NLP Research Scientist':       ['machine-learning'],
    'NLP Engineer':                 ['machine-learning'],
    'Data Scientist':               ['data-analysis', 'machine-learning'],
    'Data Analyst':                 ['data-analysis'],
    'Database Administrator':       ['data-management'],
    'Cloud Solutions Architect':    ['cloud-computing'],
    'Web Developer':                ['mobile-and-web-development', 'software-development'],
    'Mobile App Developer':         ['mobile-and-web-development'],
    'Software Engineer':            ['software-development'],
    'Information Security Analyst': ['computer-security-and-networks', 'security'],
    'Security Analyst':             ['computer-security-and-networks', 'security'],
    'Ethical Hacker':               ['computer-security-and-networks', 'security'],
    'Digital Forensics Specialist': ['computer-security-and-networks', 'security'],
    'Data Privacy Specialist':      ['computer-security-and-networks', 'security'],
    'DevOps Engineer':              ['software-development', 'cloud-computing'],
    'Distributed Systems Engineer': ['software-development', 'cloud-computing'],
    'IoT Developer':                ['software-development'],
    'Embedded Software Engineer':   ['software-development'],
    'Graphics Programmer':          ['software-development', 'design-and-product'],
    'Game Developer':               ['software-development'],
    'UX Designer':                  ['design-and-product'],
    'Network Architect':            ['networking', 'computer-security-and-networks'],
    'Bioinformatician':             ['data-analysis', 'probability-and-statistics'],
    'Geospatial Analyst':           ['data-analysis'],
    'Healthcare IT Specialist':     ['data-management', 'support-and-operations'],
}

# ============================================================
#   Helper — extract specific category from Skills tag
# ============================================================

def get_specific_cat(skills_str):
    """Extract last token (specific category) from Skills tag."""
    if not isinstance(skills_str, str) or not skills_str.strip():
        return 'unknown'
    tokens = skills_str.strip().split()
    if len(tokens) < 2:
        return 'unknown'
    if tokens[-2] in KNOWN_DOMAINS:
        return tokens[-1]
    return 'unknown'

# ============================================================
#   Evaluation function
# ============================================================

def evaluate(df, pipeline_name):
    """
    Evaluates a recommendation CSV against Coursera category
    ground truth. Returns per-student and summary results.
    """
    student_results = []

    for sid, group in df.groupby('student_id'):
        career        = group['future_career'].iloc[0]
        domain        = group['interested_domain'].iloc[0]
        expected_cats = CAREER_TO_CATEGORY.get(career)

        if not expected_cats:
            continue

        top10 = group.sort_values('rank').head(10)
        hits  = []

        for _, row in top10.iterrows():
            cat = get_specific_cat(str(row['skills'])) if pd.notna(row['skills']) else 'unknown'
            hits.append(1 if cat in expected_cats else 0)

        n    = len(hits)

        # HR@10 — at least one hit
        hr10 = 1 if sum(hits) > 0 else 0

        # CHR@10 — fraction of top-10 that are hits
        chr10 = sum(hits) / n

        # P@5 — fraction of top-5 that are hits
        p5 = sum(hits[:5]) / min(5, n)

        # MRR — reciprocal rank of first hit
        mrr = next((1/(i+1) for i, h in enumerate(hits) if h), 0.0)

        # nDCG@10 — position-weighted relevance
        dcg  = sum(h / np.log2(i+2) for i, h in enumerate(hits))
        idcg = sum(h / np.log2(i+2) for i, h in enumerate(sorted(hits, reverse=True)))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        student_results.append({
            'pipeline':          pipeline_name,
            'student_id':        sid,
            'future_career':     career,
            'interested_domain': domain,
            'expected_cats':     ', '.join(expected_cats),
            'hits':              sum(hits),
            'total_recs':        n,
            'hr@10':             hr10,
            'chr@10':            round(chr10, 4),
            'p@5':               round(p5, 4),
            'mrr':               round(mrr, 4),
            'ndcg@10':           round(ndcg, 4),
        })

    return pd.DataFrame(student_results)

# ============================================================
#   Run evaluation across all pipelines
# ============================================================

print('=' * 65)
print('EVALUATION — All Pipelines — Coursera Category Ground Truth')
print('=' * 65)

os.makedirs(OUT, exist_ok=True)

all_student_results = []
summary_rows        = []
career_rows         = []

for pipeline_name, filename in PIPELINES.items():
    path = os.path.join(RECO, filename)

    if not os.path.exists(path):
        print(f'\n[SKIP] File not found: {filename}')
        continue

    print(f'\nEvaluating: {pipeline_name}')
    df      = pd.read_csv(path, comment='#', encoding='latin-1')
    results = evaluate(df, pipeline_name)
    all_student_results.append(results)

    # Pipeline summary
    summary_rows.append({
        'Pipeline':  pipeline_name,
        'Students':  len(results),
        'HR@10':     round(results['hr@10'].mean(), 4),
        'CHR@10':    round(results['chr@10'].mean(), 4),
        'P@5':       round(results['p@5'].mean(), 4),
        'MRR':       round(results['mrr'].mean(), 4),
        'nDCG@10':   round(results['ndcg@10'].mean(), 4),
    })

    # Per-career summary for this pipeline
    career_summary = results.groupby('future_career').agg(
        students  = ('student_id', 'count'),
        hr10      = ('hr@10',  'mean'),
        chr10     = ('chr@10', 'mean'),
        p5        = ('p@5',    'mean'),
        mrr       = ('mrr',    'mean'),
        ndcg      = ('ndcg@10','mean'),
    ).reset_index()
    career_summary['pipeline'] = pipeline_name
    career_rows.append(career_summary)

    print(f'  Students : {len(results)}')
    print(f'  HR@10    : {results["hr@10"].mean():.4f}')
    print(f'  CHR@10   : {results["chr@10"].mean():.4f}')
    print(f'  P@5      : {results["p@5"].mean():.4f}')
    print(f'  MRR      : {results["mrr"].mean():.4f}')
    print(f'  nDCG@10  : {results["ndcg@10"].mean():.4f}')

# ============================================================
#   Save outputs
# ============================================================

# Per-student results
all_results_df = pd.concat(all_student_results, ignore_index=True)
all_results_df.to_csv(os.path.join(OUT, 'eval_results.csv'), index=False)

# Pipeline summary
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUT, 'eval_summary.csv'), index=False)

# Per-career results
career_df = pd.concat(career_rows, ignore_index=True)
career_df.to_csv(os.path.join(OUT, 'eval_career.csv'), index=False)

# ============================================================
#   Print final comparison table
# ============================================================

print('\n' + '=' * 65)
print('FINAL COMPARISON TABLE')
print('=' * 65)
print(summary_df.to_string(index=False))

print('\n=== WINNER PER METRIC ===')
for m in ['HR@10', 'CHR@10', 'P@5', 'MRR', 'nDCG@10']:
    winner = summary_df.loc[summary_df[m].idxmax(), 'Pipeline']
    val    = summary_df[m].max()
    print(f'  {m:<10} → {winner} ({val:.4f})')

print(f'\nOutputs saved to: {OUT}')
print('  - eval_results.csv   (per-student)')
print('  - eval_summary.csv   (per-pipeline)')
print('  - eval_career.csv    (per-career)')