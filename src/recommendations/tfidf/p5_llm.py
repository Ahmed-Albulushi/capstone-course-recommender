# ============================================================
# Pipeline 8 — LLM Profile → TF-IDF (Unfiltered)
# ============================================================
# Query construction:
#   Career title + O*NET description + SFIA descriptions
#   → LLM (claude-haiku-4-5-20251001) distils into a
#     focused 80-120 word plain-text learning profile
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
# Input:
#   - datasets/cleaned/cs_students_excluded_careers.csv
#   - datasets/cleaned/Coursera_cleaned.csv
#   - datasets/cleaned/onet_occupation_data.xlsx
#   - datasets/cleaned/sfia_standard.csv
#
# Output:
#   - results/recommendations/tfidf/unfiltered/p8_recommendations.csv
#   - results/recommendations/tfidf/unfiltered/p8_profiles.csv
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
OUT  = os.path.join(BASE, 'results', 'recommendations', 'tfidf', 'unfiltered')

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
#   Career mappings
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

CAREER_SFIA_CODES = {
    'Web Developer':                ['PROG', 'DESN'],
    'Information Security Analyst': ['SCTY', 'VUAS', 'SCAD'],
    'Mobile App Developer':         ['PROG', 'SWDN'],
    'Database Administrator':       ['DBDS', 'DBAD', 'DATM'],
    'Cloud Solutions Architect':    ['ARCH', 'IFDN'],
    'Software Engineer':            ['PROG', 'SLEN'],
    'Machine Learning Engineer':    ['MLNG', 'DATS'],
    'NLP Research Scientist':       ['MLNG', 'RSCH'],
    'Graphics Programmer':          ['ADEV', 'PROG'],
    'Data Scientist':               ['DATS', 'DAAN', 'MLNG'],
    'Data Analyst':                 ['DAAN', 'BINT', 'VISL'],
    'AI Researcher':                ['MLNG', 'RSCH', 'AIDE'],
    'Bioinformatician':             ['SCMO', 'DATS'],
    'UX Designer':                  ['HCEV', 'UNAN', 'URCH'],
    'Machine Learning Researcher':  ['MLNG', 'RSCH'],
    'Security Analyst':             ['SCTY', 'VUAS'],
    'Embedded Software Engineer':   ['RESD', 'PROG'],
    'Ethical Hacker':               ['PENT', 'VUAS'],
    'Computer Vision Engineer':     ['MLNG', 'DATS'],
    'DevOps Engineer':              ['DEPL', 'RELM', 'CFMG'],
    'IoT Developer':                ['RESD', 'NTDS'],
    'NLP Engineer':                 ['MLNG', 'DATS'],
    'Data Privacy Specialist':      ['PEDP', 'INAS'],
    'Geospatial Analyst':           ['DAAN', 'SCMO'],
    'Distributed Systems Engineer': ['NTDS', 'DESN'],
    'Digital Forensics Specialist': ['DGFS', 'CRIM'],
    'Game Developer':               ['ADEV', 'PROG'],
    'Healthcare IT Specialist':     ['DBAD', 'SCTY'],
}

# ============================================================
#   Step 1 — Load datasets
# ============================================================

students = pd.read_csv(os.path.join(DATA, 'cs_students_excluded_careers.csv'))
courses  = pd.read_csv(os.path.join(DATA, 'Coursera_cleaned.csv'))
onet     = pd.read_excel(os.path.join(DATA, 'onet_occupation_data.xlsx'))
sfia     = pd.read_csv(os.path.join(DATA, 'sfia_standard.csv'), encoding='latin1')

# Normalise course rating
courses['Course Rating'] = pd.to_numeric(courses['Course Rating'], errors='coerce')
median_rating            = courses['Course Rating'].median()
courses['Course Rating'] = courses['Course Rating'].fillna(median_rating)
courses['rating_norm']   = courses['Course Rating'] / 5.0

print('=' * 65)
print('PIPELINE 8 — LLM Profile → TF-IDF (Unfiltered)')
print('=' * 65)
print(f'Students : {len(students)}')
print(f'Courses  : {len(courses)}')
print(f'Top N    : {TOP_N}')

# ============================================================
#   Step 2 — Description retrieval functions
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

def get_sfia_description(codes, sfia_df):
    all_text = []
    for code in codes:
        matches = sfia_df[sfia_df['Code'] == code]
        if matches.empty:
            continue
        row   = matches.iloc[0]
        parts = [str(row[c]) for c in [
            'Overall description', 'Guidance notes',
            'Level 1 description', 'Level 2 description',
            'Level 3 description', 'Level 4 description',
            'Level 5 description', 'Level 6 description',
            'Level 7 description'
        ]]
        all_text.append(' '.join([p for p in parts if p != 'nan']))
    return ' '.join(all_text)

# ============================================================
#   Step 3 — LLM profile generation
#            Strips markdown programmatically as safety net
# ============================================================

def clean_profile(text):
    """Remove markdown headers, bullets, and extra whitespace."""
    text = re.sub(r'^#+\s.*$', '', text, flags=re.MULTILINE)  # headers
    text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)  # bullets
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # bold
    text = re.sub(r'\*(.+?)\*', r'\1', text)       # italic
    text = re.sub(r'\n{2,}', ' ', text)             # multiple newlines
    return text.strip()

def generate_profile(career, onet_desc, sfia_desc):
    """
    Calls claude-haiku-4-5-20251001 to distil Career + O*NET + SFIA
    into a focused 80-120 word plain-text learning interest profile.
    LLM acts as compression step only — no external knowledge added.
    """
    source = f"""Career: {career}

O*NET Description:
{onet_desc[:1500]}

SFIA Skill Descriptions:
{sfia_desc[:1500]}"""

    prompt = f"""You are helping build a course recommendation system for CS students.

Based ONLY on the source data below, write a learning interest profile for a student 
who wants to become a {career}.

STRICT RULES — you must follow all of these:
- Plain paragraph only — absolutely NO headers, NO bullet points, NO bold, NO markdown
- 80 to 120 words exactly
- Use only information from the source data — do not add external knowledge
- Write in third person as learning goals (e.g. "The student wants to learn...")
- Focus on specific tools, technologies, and techniques

Source data:
{source}"""

    payload = {
        'model':      'claude-haiku-4-5-20251001',
        'max_tokens': 200,
        'messages':   [{'role': 'user', 'content': prompt}],
    }

    for attempt in range(3):
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        if response.status_code == 200:
            data = response.json()
            raw  = data['content'][0]['text'].strip()
            return clean_profile(raw)
        elif response.status_code == 529:
            print(f'    [RETRY {attempt+1}/3] API overloaded, waiting 10s...')
            time.sleep(10)
        else:
            print(f'    [ERROR] API returned {response.status_code}: {response.text}')
            break
    return ''



# ============================================================
#   Step 4 — Generate LLM profiles (one per unique career)
# ============================================================

print('\nGenerating LLM profiles...')
unique_careers = students['Future Career'].unique()
print(f'Unique careers: {len(unique_careers)}')

profile_cache = {}
profile_rows  = []
skipped       = []

for career in unique_careers:
    onet_desc = get_onet_description(career, onet, CAREER_ONET_ROW)
    sfia_desc = get_sfia_description(CAREER_SFIA_CODES.get(career, []), sfia)

    if not onet_desc and not sfia_desc:
        print(f'  [SKIP] No source data for: {career}')
        skipped.append(career)
        continue

    print(f'  Generating: {career}')
    profile = generate_profile(career, onet_desc, sfia_desc)
    profile_cache[career] = profile
    profile_rows.append({
        'career':        career,
        'profile':       profile,
        'word_count':    len(profile.split()),
        'onet_chars':    len(onet_desc),
        'sfia_chars':    len(sfia_desc),
    })
    time.sleep(0.3)

# Save profiles for inspection
os.makedirs(OUT, exist_ok=True)
profiles_df = pd.DataFrame(profile_rows)
profiles_df.to_csv(os.path.join(OUT, 'p5_profiles.csv'), index=False)
print(f'\nProfiles saved → p8_profiles.csv')
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
#   Step 7 — Run for all students
# ============================================================

all_recs     = []
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

out_path   = os.path.join(OUT, 'p5_recommendations.csv')
results_df = pd.DataFrame(all_recs)

with open(out_path, 'w') as f:
    f.write('# Pipeline      : P8 — LLM Profile → TF-IDF (Unfiltered)\n')
    f.write('# Query         : LLM-distilled profile from Career + O*NET + SFIA\n')
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