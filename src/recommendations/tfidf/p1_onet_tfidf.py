# ============================================================
# Pipeline 1 — Career + O*NET → TF-IDF
# ============================================================
# Query construction:
#   Career title + O*NET occupation description only
#   No SFIA — isolates the O*NET contribution
#
# Retrieval:
#   TF-IDF vectorisation + cosine similarity
#   Library: scikit-learn (Pedregosa et al., 2011)
#   Vocabulary: full course corpus (3,424 courses)
#
# Re-ranking:
#   final_score = sim_score × (course_rating / 5.0)
#
# Domain filter: None
#
# Course text: Course Name + Course Description ONLY
#   Skills tags excluded — reserved for evaluation
#
# O*NET mapping: 20 careers with exact or close matches only
#   13 careers excluded due to missing or misleading O*NET rows
#   (see preprocess_students.py for full exclusion justification)
#
# Comparison:
#   P1 vs P3 → effect of LLM on career-title-only query
#   P1 vs P4 → effect of LLM on O*NET query
#
# Input:
#   - datasets/cleaned/cs_students_excluded_careers.csv
#   - datasets/cleaned/Coursera_cleaned.csv
#   - datasets/cleaned/onet_occupation_data.xlsx
#
# Output:
#   - results/recommendations/tfidf/p1_recommendations.csv
# ============================================================

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = '/Users/soesoe/Documents/Capstone Project/final_capstone-course-recommender'
DATA = os.path.join(BASE, 'datasets', 'cleaned')
OUT  = os.path.join(BASE, 'results', 'recommendations', 'tfidf')

TOP_N     = 10
MAX_FEATS = 20000

# ============================================================
#   Final O*NET mapping — 20 careers
#   Only exact and close matches retained for fairness
# ============================================================

CAREER_ONET_ROW = {
    'Information Security Analyst': 113,
    'Security Analyst':             113,
    'Machine Learning Researcher':  114,
    'AI Researcher':                114,
    'NLP Research Scientist':       114,
    'Cloud Solutions Architect':    117,
    'Database Administrator':       119,
    'Software Engineer':            124,
    'Web Developer':                126,
    'UX Designer':                  127,
    'Game Developer':               128,
    'Geospatial Analyst':           131,
    'Ethical Hacker':               133,
    'Digital Forensics Specialist': 135,
    'Distributed Systems Engineer': 137,
    'Data Scientist':               144,
    'Machine Learning Engineer':    144,
    'NLP Engineer':                 144,
    'Data Analyst':                 145,
    'Bioinformatician':             215,
}

students = pd.read_csv(os.path.join(DATA, 'cs_students_excluded_careers.csv'))
courses  = pd.read_csv(os.path.join(DATA, 'Coursera_cleaned.csv'))
onet     = pd.read_excel(os.path.join(DATA, 'onet_occupation_data.xlsx'))

courses['Course Rating'] = pd.to_numeric(courses['Course Rating'], errors='coerce')
courses['Course Rating'] = courses['Course Rating'].fillna(courses['Course Rating'].median())
courses['rating_norm']   = courses['Course Rating'] / 5.0

print('=' * 65)
print('PIPELINE 1 — Career + O*NET → TF-IDF')
print('=' * 65)
print(f'Students : {len(students)}')
print(f'Courses  : {len(courses)}')
print(f'Top N    : {TOP_N}')

def get_onet_description(career, onet_df, career_map):
    row_val = career_map.get(career)
    if row_val is None:
        return ''
    return onet_df.iloc[row_val - 2]['Description']

course_texts = (
    courses['Course Name'].fillna('') + ' ' +
    courses['Course Description'].fillna('')
).tolist()

print(f'\nCourse corpus built: {len(course_texts)} documents')

def recommend_tfidf(query, course_corpus, courses_df, top_n=TOP_N):
    corpus  = [query] + course_corpus
    vec     = TfidfVectorizer(stop_words='english', max_features=MAX_FEATS)
    tfidf   = vec.fit_transform(corpus)
    scores  = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    top_idx = scores.argsort()[-top_n:][::-1]
    return [{'course': courses_df.iloc[i]['Course Name'],
             'sim_score': round(float(scores[i]), 4),
             'rating': courses_df.iloc[i]['Course Rating'],
             'rating_norm': round(courses_df.iloc[i]['rating_norm'], 4),
             'level': courses_df.iloc[i]['Difficulty Level'],
             'skills': courses_df.iloc[i]['Skills']} for i in top_idx]

def rerank(matches):
    for c in matches:
        c['final_score'] = round(c['sim_score'] * c['rating_norm'], 4)
    return sorted(matches, key=lambda x: x['final_score'], reverse=True)

all_recs     = []
skipped      = []
career_cache = {}

print('\nGenerating recommendations...')

for _, student in students.iterrows():
    career = student['Future Career']
    if career not in career_cache:
        onet_desc = get_onet_description(career, onet, CAREER_ONET_ROW)
        if not onet_desc:
            skipped.append(career)
            career_cache[career] = None
            continue
        query                = f"{career} {onet_desc}"
        ranked               = rerank(recommend_tfidf(query, course_texts, courses))
        career_cache[career] = ranked
        print(f'  Computed: {career}')

    ranked = career_cache.get(career)
    if not ranked:
        continue

    for rank, c in enumerate(ranked[:TOP_N], 1):
        all_recs.append({
            'student_id': student['Student ID'], 'student_name': student['Name'],
            'future_career': career, 'interested_domain': student['Interested Domain'],
            'rank': rank, 'course': c['course'], 'rating': c['rating'],
            'rating_norm': c['rating_norm'], 'level': c['level'], 'skills': c['skills'],
            'sim_score': c['sim_score'], 'final_score': c['final_score'],
        })

os.makedirs(OUT, exist_ok=True)
out_path   = os.path.join(OUT, 'p1_recommendations.csv')
results_df = pd.DataFrame(all_recs)

with open(out_path, 'w') as f:
    f.write('# Pipeline      : P1 — Career + O*NET → TF-IDF\n')
    f.write('# Query         : Career title + O*NET occupation description\n')
    f.write('# Retrieval     : TF-IDF cosine similarity\n')
    f.write('# Domain filter : None\n')
    f.write('# SFIA          : Not used\n')
    f.write('# Students      : cs_students_excluded_careers.csv\n')
    f.write('# Courses       : Coursera_cleaned.csv\n')
    f.write(f'# Total recs    : {len(results_df)}\n')
    f.write(f'# Students cov  : {results_df["student_id"].nunique()}\n')
    f.write(f'# Careers cov   : {results_df["future_career"].nunique()}\n')
    f.write('#\n')
results_df.to_csv(out_path, mode='a', index=False)

print(f'\nTotal recommendations : {len(results_df)}')
print(f'Students covered      : {results_df["student_id"].nunique()}')
print(f'Careers covered       : {results_df["future_career"].nunique()}')
if skipped: print(f'Skipped               : {set(skipped)}')
print(f'\nSaved → {out_path}')
