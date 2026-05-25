# ============================================================
# Evaluation Script — Excluded Careers (P3 only)
# ============================================================
# Purpose:
#   Tests Pipeline 3 (LLM career-only) on careers that were
#   excluded from the main evaluation due to missing or
#   misleading O*NET mappings.
#
#   Since P3 uses only the career title (no O*NET), it is
#   not affected by the O*NET mapping issues that caused
#   exclusion. This provides insight into whether the system
#   can still recommend relevant courses for these careers.
#
# Excluded careers tested:
#   - Mobile App Developer      (no O*NET match)
#   - Computer Vision Engineer  (no O*NET match)
#   - DevOps Engineer           (no O*NET match)
#   - IoT Developer             (no O*NET match)
#   - Embedded Software Engineer(no O*NET match)
#   - Data Privacy Specialist   (wrong O*NET row)
#   - Healthcare IT Specialist  (wrong O*NET row)
#   - Graphics Programmer       (wrong O*NET row)
#   - Data Analyst              (no valid O*NET match)
#   - Game Developer            (no valid O*NET match)
#   Note: Quantum Computing, VR, Robotics, Blockchain, SEO
#   excluded — no clear Coursera category match
#
# Ground truth: Coursera category labels (same as main eval)
#
# Input:
#   - datasets/cleaned/cs_students_cleaned.csv (full dataset)
#   - results/recommendations/tfidf/p3_excluded_recommendations.csv
#
# Output:
#   - results/evaluation/eval_excluded_summary.csv
#   - results/evaluation/eval_excluded_career.csv
# ============================================================

import os
import pandas as pd
import numpy as np

BASE = '/Users/soesoe/Documents/Capstone Project/final_capstone-course-recommender'
RECO = os.path.join(BASE, 'results', 'recommendations', 'tfidf')
OUT  = os.path.join(BASE, 'results', 'evaluation')

KNOWN_DOMAINS = {
    'business','computer-science','data-science','life-sciences',
    'physical-science-and-engineering','social-sciences','arts-and-humanities',
    'information-technology','language-learning','personal-development','math-and-logic'
}

# ============================================================
#   Excluded careers — Coursera category mapping
#   Careers with no clear Coursera category are omitted
# ============================================================

EXCLUDED_CAREER_TO_CATEGORY = {
    'Mobile App Developer':      ['mobile-and-web-development'],
    'Computer Vision Engineer':  ['machine-learning'],
    'DevOps Engineer':           ['cloud-computing', 'software-development'],
    'IoT Developer':             ['software-development'],
    'Embedded Software Engineer':['software-development'],
    'Data Privacy Specialist':   ['computer-security-and-networks', 'security'],
    'Healthcare IT Specialist':  ['data-management', 'support-and-operations'],
    'Graphics Programmer':       ['software-development', 'design-and-product'],
    'Data Analyst':              ['data-analysis'],
    'Game Developer':            ['software-development'],
    # Excluded — no clear Coursera category:
    # Quantum Computing Researcher, VR Developer, Robotics Engineer,
    # Blockchain Engineer, SEO Specialist
}

def get_specific_cat(skills_str):
    if not isinstance(skills_str, str) or not skills_str.strip():
        return 'unknown'
    tokens = skills_str.strip().split()
    if len(tokens) < 2:
        return 'unknown'
    if tokens[-2] in KNOWN_DOMAINS:
        return tokens[-1]
    return 'unknown'

def evaluate(df):
    results = []
    for sid, group in df.groupby('student_id'):
        career        = group['future_career'].iloc[0]
        expected_cats = EXCLUDED_CAREER_TO_CATEGORY.get(career)
        if not expected_cats:
            continue
        top10 = group.sort_values('rank').head(10)
        hits  = []
        for _, row in top10.iterrows():
            cat = get_specific_cat(str(row['skills'])) if pd.notna(row['skills']) else 'unknown'
            hits.append(1 if cat in expected_cats else 0)
        n    = len(hits)
        hr10 = 1 if sum(hits) > 0 else 0
        chr10= sum(hits)/n
        p5   = sum(hits[:5])/min(5,n)
        mrr  = next((1/(i+1) for i,h in enumerate(hits) if h), 0.0)
        dcg  = sum(h/np.log2(i+2) for i,h in enumerate(hits))
        idcg = sum(h/np.log2(i+2) for i,h in enumerate(sorted(hits,reverse=True)))
        ndcg = dcg/idcg if idcg>0 else 0.0
        results.append(dict(
            student_id=sid, career=career,
            expected_cats=', '.join(expected_cats),
            hits=sum(hits), total_recs=n,
            hr10=hr10, chr10=round(chr10,4),
            p5=round(p5,4), mrr=round(mrr,4), ndcg=round(ndcg,4)
        ))
    return pd.DataFrame(results)

print('=' * 65)
print('EVALUATION — Excluded Careers — P3 (LLM career only)')
print('=' * 65)

path = os.path.join(RECO, 'p3_excluded_recommendations.csv')
if not os.path.exists(path):
    print(f'\n[ERROR] File not found: p3_excluded_recommendations.csv')
    print('Please run p3_excluded.py first to generate recommendations')
    print('for the excluded careers.')
    exit()

df      = pd.read_csv(path, comment='#', encoding='latin-1')
results = evaluate(df)

print(f'Students evaluated : {len(results)}')
print(f'Careers evaluated  : {results["career"].nunique()}')

print(f'\nOverall metrics:')
print(f'  HR@10   : {results["hr10"].mean():.4f}')
print(f'  CHR@10  : {results["chr10"].mean():.4f}')
print(f'  P@5     : {results["p5"].mean():.4f}')
print(f'  MRR     : {results["mrr"].mean():.4f}')
print(f'  nDCG@10 : {results["ndcg"].mean():.4f}')

career_summary = results.groupby('career').agg(
    students=('student_id','count'),
    hr10=('hr10','mean'), chr10=('chr10','mean'),
    p5=('p5','mean'), mrr=('mrr','mean'), ndcg=('ndcg','mean')
).reset_index().sort_values('mrr', ascending=False)

print(f'\nPer-career results:')
print(f"{'Career':<30} {'N':>4} {'HR@10':>6} {'CHR@10':>7} {'P@5':>6} {'MRR':>6} {'nDCG':>6}")
print('-' * 70)
for _, row in career_summary.iterrows():
    print(f"{row['career']:<30} {int(row['students']):>4} {row['hr10']:>6.3f} {row['chr10']:>7.3f} {row['p5']:>6.3f} {row['mrr']:>6.3f} {row['ndcg']:>6.3f}")

os.makedirs(OUT, exist_ok=True)
career_summary.to_csv(os.path.join(OUT, 'eval_excluded_career.csv'), index=False)
results.to_csv(os.path.join(OUT, 'eval_excluded_results.csv'), index=False)
print(f'\nSaved → eval_excluded_career.csv')
print(f'Saved → eval_excluded_results.csv')