# ============================================================
# Pipeline 5 — Career + O*NET → LLM → TF-IDF
# ============================================================
# Query construction:
#   Career title + O*NET occupation description →
#   LLM (claude-haiku-4-5-20251001) distils into a
#   focused 80-120 word plain-text learning profile
#   One API call per unique career (28 calls total)
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
# LLM prompt: identical instruction block to P3 and P4
#   Only the source data section differs — ensuring fair
#   comparison across LLM pipelines
#
# Comparison:
#   P5 vs P3 → effect of adding O*NET to LLM prompt
#   P5 vs P4 → effect of adding SFIA to O*NET+LLM prompt
#   P5 vs P1 → effect of LLM distillation on O*NET query
#
# Input:
#   - datasets/cleaned/cs_students_excluded_careers.csv
#   - datasets/cleaned/Coursera_cleaned.csv
#   - datasets/cleaned/onet_occupation_data.xlsx
#
# Output:
#   - results/recommendations/tfidf/p5_recommendations.csv
#   - results/recommendations/tfidf/p5_profiles.csv
# ============================================================

import os
import re
import time
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
#   Paths
# ============================================================

BASE = '/Users/soesoe/Documents/Capstone Project/final_capstone-course-recommender'
DATA = os.path.join(BASE, 'datasets', 'cleaned')
OUT  = os.path.join(BASE, 'results', 'recommendations', 'tfidf')

# ============================================================
#   Configuration
# ============================================================

TOP_N     = 10
MAX_FEATS = 20000

API_KEY = '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'
API_URL = 'https://api.anthropic.com/v1/messages'
HEADERS = {
    'x-api-key':         API_KEY,
    'anthropic-version': '2023-06-01',
    'content-type':      'application/json',
}

# ============================================================
#   Career → O*NET row mapping
# ============================================================

CAREER_ONET_ROW = {
    'Web Developer':                126,
    'Information Security Analyst': 113,
    'Mobile App Developer':         124,
    'Database Administrator':       119,
    'Cloud Solutions Architect':    137,
    'Software Engineer':            124,
    'Machine Learning Engineer':    144,
    'NLP Research Scientist':       114,
    'Graphics Programmer':          123,
    'Data Scientist':               144,
    'Data Analyst':                 145,
    'AI Researcher':                114,
    'Bioinformatician':             215,
    'UX Designer':                  127,
    'Machine Learning Researcher':  'combined_114_144',
    'Security Analyst':             113,
    'Embedded Software Engineer':   182,
    'Ethical Hacker':               133,
    'Computer Vision Engineer':     114,
    'DevOps Engineer':              124,
    'IoT Developer':                'combined_124_182',
    'NLP Engineer':                 144,
    'Data Privacy Specialist':      134,
    'Geospatial Analyst':           131,
    'Distributed Systems Engineer': 117,
    'Digital Forensics Specialist': 135,
    'Game Developer':               123,
    'Healthcare IT Specialist':     119,
}

# ============================================================
#   Step 1 — Load datasets
# ============================================================

students = pd.read_csv(os.path.join(DATA, 'cs_students_excluded_careers.csv'))
courses  = pd.read_csv(os.path.join(DATA, 'Coursera_cleaned.csv'))
onet     = pd.read_excel(os.path.join(DATA, 'onet_occupation_data.xlsx'))

# Normalise course rating
courses['Course Rating'] = pd.to_numeric(courses['Course Rating'], errors='coerce')
median_rating            = courses['Course Rating'].median()
courses['Course Rating'] = courses['Course Rating'].fillna(median_rating)
courses['rating_norm']   = courses['Course Rating'] / 5.0

print('=' * 65)
print('PIPELINE 5 — Career + O*NET → LLM → TF-IDF')
print('=' * 65)
print(f'Students : {len(students)}')
print(f'Courses  : {len(courses)}')
print(f'Top N    : {TOP_N}')

# ============================================================
#   Step 2 — O*NET description retrieval
# ============================================================

def get_onet_description(career, onet_df, career_map):
    row_val = career_map.get(career)
    if row_val is None:
        return ''
    if row_val == 'combined_114_144':
        return onet_df.iloc[112]['Description'] + ' ' + onet_df.iloc[142]['Description']
    if row_val == 'combined_124_182':
        return onet_df.iloc[122]['Description'] + ' ' + onet_df.iloc[180]['Description']
    return onet_df.iloc[row_val - 2]['Description']

# ============================================================
#   Step 3 — LLM profile generation
#            Identical instruction block to P3 and P4
#            Only source data section differs
# ============================================================

def clean_profile(text):
    """Remove any markdown formatting as safety net."""
    text = re.sub(r'^#+\s.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'\n{2,}', ' ', text)
    return text.strip()

def generate_profile(career, onet_desc):
    """
    Calls LLM with career title + O*NET description only.
    LLM distils structured O*NET data — no SFIA provided.
    Prompt instruction block identical to P3 and P4.
    """
    source = f"""Career: {career}

O*NET Occupation Description:
{onet_desc[:2000]}"""

    prompt = f"""You are helping build a course recommendation system for CS students.

Based ONLY on the source data below, write a learning interest profile for a student
who wants to become a {career}.

STRICT RULES — you must follow all of these:
- Plain paragraph only — absolutely NO headers, NO bullet points, NO bold, NO markdown
- 80 to 120 words exactly
- Use only information from the source data — do not add external knowledge
- Write in third person as learning goals (e.g. "The student wants to learn...")
- Focus on specific tools, technologies, techniques, and skills

Source data:
{source}"""

    payload = {
        'model':      'claude-haiku-4-5-20251001',
        'max_tokens': 200,
        'messages':   [{'role': 'user', 'content': prompt}],
    }

    for attempt in range(5):
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        if response.status_code == 200:
            data = response.json()
            raw  = data['content'][0]['text'].strip()
            return clean_profile(raw)
        elif response.status_code == 529:
            print(f'    [RETRY {attempt+1}/5] API overloaded, waiting 30s...')
            time.sleep(30)
        else:
            print(f'    [ERROR] API returned {response.status_code}: {response.text}')
            break
    return ''

# ============================================================
#   Step 4 — Generate profiles (one per unique career)
# ============================================================

print('\nGenerating LLM profiles...')
unique_careers = students['Future Career'].unique()
print(f'Unique careers: {len(unique_careers)}')

profile_cache = {}
profile_rows  = []
skipped       = []

for career in unique_careers:
    onet_desc = get_onet_description(career, onet, CAREER_ONET_ROW)

    if not onet_desc:
        print(f'  [SKIP] No O*NET data for: {career}')
        skipped.append(career)
        continue

    print(f'  Generating: {career}')
    profile = generate_profile(career, onet_desc)

    if not profile:
        print(f'  [SKIP] Failed to generate profile for: {career}')
        skipped.append(career)
        continue

    profile_cache[career] = profile
    profile_rows.append({
        'career':     career,
        'profile':    profile,
        'word_count': len(profile.split()),
        'onet_chars': len(onet_desc),
    })
    time.sleep(0.3)

# Save profiles
os.makedirs(OUT, exist_ok=True)
profiles_df = pd.DataFrame(profile_rows)
profiles_df.to_csv(os.path.join(OUT, 'p4_profiles.csv'), index=False)
print(f'\nProfiles saved → p4_profiles.csv')
print(f'Avg word count: {profiles_df["word_count"].mean():.0f} words')

# ============================================================
#   Step 5 — Build course corpus
# ============================================================

course_texts = (
    courses['Course Name'].fillna('') + ' ' +
    courses['Course Description'].fillna('')
).tolist()

print(f'\nCourse corpus built: {len(course_texts)} documents')

# ============================================================
#   Step 6 — TF-IDF matching function
# ============================================================

def recommend_tfidf(query, course_corpus, courses_df, top_n=TOP_N):
    corpus  = [query] + course_corpus
    vec     = TfidfVectorizer(stop_words='english', max_features=MAX_FEATS)
    tfidf   = vec.fit_transform(corpus)
    scores  = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    top_idx = scores.argsort()[-top_n:][::-1]

    return [
        {
            'course':      courses_df.iloc[i]['Course Name'],
            'sim_score':   round(float(scores[i]), 4),
            'rating':      courses_df.iloc[i]['Course Rating'],
            'rating_norm': round(courses_df.iloc[i]['rating_norm'], 4),
            'level':       courses_df.iloc[i]['Difficulty Level'],
            'skills':      courses_df.iloc[i]['Skills'],
        }
        for i in top_idx
    ]

def rerank(matches):
    for c in matches:
        c['final_score'] = round(c['sim_score'] * c['rating_norm'], 4)
    return sorted(matches, key=lambda x: x['final_score'], reverse=True)

# ============================================================
#   Step 7 — Run for all students (cached per career)
# ============================================================

all_recs          = []
career_cache_recs = {}

print('\nGenerating recommendations...')

for _, student in students.iterrows():
    career  = student['Future Career']
    profile = profile_cache.get(career)

    if not profile:
        continue

    if career not in career_cache_recs:
        ranked = rerank(recommend_tfidf(profile, course_texts, courses))
        career_cache_recs[career] = ranked
        print(f'  Computed: {career}')

    ranked = career_cache_recs[career]
    for rank, c in enumerate(ranked[:TOP_N], 1):
        all_recs.append({
            'student_id':        student['Student ID'],
            'student_name':      student['Name'],
            'future_career':     career,
            'interested_domain': student['Interested Domain'],
            'rank':              rank,
            'course':            c['course'],
            'rating':            c['rating'],
            'rating_norm':       c['rating_norm'],
            'level':             c['level'],
            'skills':            c['skills'],
            'sim_score':         c['sim_score'],
            'final_score':       c['final_score'],
        })

# ============================================================
#   Step 8 — Save with metadata header
# ============================================================

out_path   = os.path.join(OUT, 'p4_recommendations.csv')
results_df = pd.DataFrame(all_recs)

with open(out_path, 'w') as f:
    f.write('# Pipeline      : P5 — Career + O*NET → LLM → TF-IDF\n')
    f.write('# Query         : LLM-distilled profile from Career title + O*NET only\n')
    f.write('# LLM source    : O*NET occupation description (no SFIA)\n')
    f.write('# Retrieval     : TF-IDF cosine similarity\n')
    f.write('# Domain filter : None\n')
    f.write('# LLM model     : claude-haiku-4-5-20251001\n')
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
if skipped:
    print(f'Skipped               : {set(skipped)}')
print(f'\nSaved → {out_path}')