# ============================================================
# Pipeline 3 — Excluded Careers (Career Title → LLM → TF-IDF)
# ============================================================
# Purpose:
#   Generates recommendations for students whose careers were
#   excluded from the main evaluation due to missing or
#   misleading O*NET mappings.
#
#   Since P3 uses only the career title (no O*NET), it is
#   not affected by O*NET mapping issues and can generate
#   recommendations for all careers.
#
#   This is a supplementary analysis — not part of the main
#   pipeline comparison. Results reported separately.
#
# Excluded careers included:
#   Mobile App Developer, Computer Vision Engineer,
#   DevOps Engineer, IoT Developer, Embedded Software Engineer,
#   Data Privacy Specialist, Healthcare IT Specialist,
#   Graphics Programmer, Data Analyst, Game Developer
#   (Quantum Computing, VR, Robotics, Blockchain, SEO also
#   included in reco but excluded from eval — no Coursera match)
#
# Input:
#   - datasets/cleaned/cs_students_cleaned.csv (full dataset)
#   - datasets/cleaned/Coursera_cleaned.csv
#
# Output:
#   - results/recommendations/tfidf/p3_excluded_recommendations.csv
#   - results/recommendations/tfidf/p3_excluded_profiles.csv
# ============================================================

import os
import re
import time
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = '/Users/soesoe/Documents/Capstone Project/final_capstone-course-recommender'
DATA = os.path.join(BASE, 'datasets', 'cleaned')
OUT  = os.path.join(BASE, 'results', 'recommendations', 'tfidf')

TOP_N     = 10
MAX_FEATS = 20000

API_KEY = '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'
API_URL = 'https://api.anthropic.com/v1/messages'
HEADERS = {
    'x-api-key': API_KEY,
    'anthropic-version': '2023-06-01',
    'content-type': 'application/json',
}

# ============================================================
#   Excluded careers — to recommend for
# ============================================================

EXCLUDED_CAREERS = {
    'Mobile App Developer', 'Computer Vision Engineer',
    'DevOps Engineer', 'IoT Developer', 'Embedded Software Engineer',
    'Data Privacy Specialist', 'Healthcare IT Specialist',
    'Graphics Programmer', 'Data Analyst', 'Game Developer',
    'Quantum Computing Researcher', 'VR Developer',
    'Robotics Engineer', 'Blockchain Engineer', 'SEO Specialist',
}

# ============================================================
#   Load datasets — full student file, filter to excluded only
# ============================================================

all_students = pd.read_csv(os.path.join(DATA, 'cs_students_cleaned.csv'))
students     = all_students[all_students['Future Career'].isin(EXCLUDED_CAREERS)].reset_index(drop=True)
courses      = pd.read_csv(os.path.join(DATA, 'Coursera_cleaned.csv'))

courses['Course Rating'] = pd.to_numeric(courses['Course Rating'], errors='coerce')
courses['Course Rating'] = courses['Course Rating'].fillna(courses['Course Rating'].median())
courses['rating_norm']   = courses['Course Rating'] / 5.0

print('=' * 65)
print('PIPELINE 3 — Excluded Careers — Career Title → LLM → TF-IDF')
print('=' * 65)
print(f'Excluded students : {len(students)}')
print(f'Unique careers    : {students["Future Career"].nunique()}')
print(f'Courses           : {len(courses)}')
print(f'\nCareers included:')
for c in sorted(students['Future Career'].unique()):
    n = len(students[students['Future Career']==c])
    print(f'  {c}: {n} students')

def clean_profile(text):
    text = re.sub(r'^#+\s.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'\n{2,}', ' ', text)
    return text.strip()

def generate_profile(career):
    prompt = f"""You are helping build a course recommendation system for CS students.

Generate a focused learning interest profile for a student who wants to become a {career}.

STRICT RULES — you must follow all of these:
- Plain paragraph only — absolutely NO headers, NO bullet points, NO bold, NO markdown
- 80 to 120 words exactly
- Write in third person as learning goals (e.g. "The student wants to learn...")
- Focus on specific tools, technologies, techniques, and skills relevant to this career"""

    payload = {
        'model': 'claude-haiku-4-5-20251001',
        'max_tokens': 200,
        'messages': [{'role': 'user', 'content': prompt}],
    }
    for attempt in range(5):
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        if response.status_code == 200:
            return clean_profile(response.json()['content'][0]['text'].strip())
        elif response.status_code == 529:
            print(f'    [RETRY {attempt+1}/5] API overloaded, waiting 30s...')
            time.sleep(30)
        else:
            print(f'    [ERROR] {response.status_code}: {response.text}')
            break
    return ''

print('\nGenerating LLM profiles...')
unique_careers = students['Future Career'].unique()

profile_cache = {}
profile_rows  = []
skipped       = []

for career in unique_careers:
    print(f'  Generating: {career}')
    profile = generate_profile(career)
    if not profile:
        skipped.append(career)
        continue
    profile_cache[career] = profile
    profile_rows.append({'career': career, 'profile': profile, 'word_count': len(profile.split())})
    time.sleep(0.3)

os.makedirs(OUT, exist_ok=True)
pd.DataFrame(profile_rows).to_csv(os.path.join(OUT, 'p3_excluded_profiles.csv'), index=False)
print(f'Profiles saved → p3_excluded_profiles.csv')

course_texts = (courses['Course Name'].fillna('') + ' ' + courses['Course Description'].fillna('')).tolist()

def recommend_tfidf(query, course_corpus, courses_df, top_n=TOP_N):
    corpus  = [query] + course_corpus
    vec     = TfidfVectorizer(stop_words='english', max_features=MAX_FEATS)
    tfidf   = vec.fit_transform(corpus)
    scores  = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    top_idx = scores.argsort()[-top_n:][::-1]
    return [{'course': courses_df.iloc[i]['Course Name'],
             'sim_score': round(float(scores[i]),4),
             'rating': courses_df.iloc[i]['Course Rating'],
             'rating_norm': round(courses_df.iloc[i]['rating_norm'],4),
             'level': courses_df.iloc[i]['Difficulty Level'],
             'skills': courses_df.iloc[i]['Skills']} for i in top_idx]

def rerank(matches):
    for c in matches:
        c['final_score'] = round(c['sim_score'] * c['rating_norm'], 4)
    return sorted(matches, key=lambda x: x['final_score'], reverse=True)

all_recs = []
career_cache_recs = {}

print('\nGenerating recommendations...')
for _, student in students.iterrows():
    career  = student['Future Career']
    profile = profile_cache.get(career)
    if not profile:
        continue
    if career not in career_cache_recs:
        career_cache_recs[career] = rerank(recommend_tfidf(profile, course_texts, courses))
        print(f'  Computed: {career}')
    for rank, c in enumerate(career_cache_recs[career][:TOP_N], 1):
        all_recs.append({
            'student_id': student['Student ID'], 'student_name': student['Name'],
            'future_career': career, 'interested_domain': student['Interested Domain'],
            'rank': rank, 'course': c['course'], 'rating': c['rating'],
            'rating_norm': c['rating_norm'], 'level': c['level'], 'skills': c['skills'],
            'sim_score': c['sim_score'], 'final_score': c['final_score'],
        })

out_path   = os.path.join(OUT, 'p3_excluded_recommendations.csv')
results_df = pd.DataFrame(all_recs)

with open(out_path, 'w') as f:
    f.write('# Pipeline      : P3 Excluded — Career Title → LLM → TF-IDF\n')
    f.write('# Purpose       : Recommendations for excluded career students\n')
    f.write('# Note          : Supplementary analysis — not in main comparison\n')
    f.write('# LLM model     : claude-haiku-4-5-20251001\n')
    f.write(f'# Total recs    : {len(results_df)}\n')
    f.write(f'# Students cov  : {results_df["student_id"].nunique()}\n')
    f.write(f'# Careers cov   : {results_df["future_career"].nunique()}\n')
    f.write('#\n')
results_df.to_csv(out_path, mode='a', index=False)

print(f'\nTotal recommendations : {len(results_df)}')
print(f'Students covered      : {results_df["student_id"].nunique()}')
print(f'Careers covered       : {results_df["future_career"].nunique()}')
if skipped: print(f'Skipped: {set(skipped)}')
print(f'\nSaved → {out_path}')