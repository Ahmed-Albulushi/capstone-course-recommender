# ============================================================
# Pipeline 7 — O*NET + SFIA → TF-IDF (Unfiltered)
# ============================================================
# Query construction:
#   Career title + O*NET occupation description +
#   SFIA full skill descriptions (all 7 levels)
#
# Retrieval:
#   TF-IDF vectorisation + cosine similarity
#   Library: scikit-learn (Pedregosa et al., 2011)
#   Vocabulary: full course corpus (3,424 courses)
#
# Re-ranking:
#   final_score = sim_score × (course_rating / 5.0)
#
# Course text: Course Name + Course Description ONLY
#   Skills tags excluded — reserved for evaluation
#
# Domain filter: None
#
# Input:
#   - datasets/cleaned/cs_students_cleaned.csv
#   - datasets/cleaned/Coursera_cleaned.csv
#   - datasets/cleaned/onet_occupation_data.xlsx
#   - datasets/cleaned/sfia_standard.csv
#
# Output:
#   - results/recommendations/tfidf/unfiltered/p7_recommendations.csv
# ============================================================

import os
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

TOP_N      = 10
MAX_FEATS  = 20000    # TF-IDF vocabulary size

# ============================================================
#   Shared constants — Career mappings
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
    'Quantum Computing Researcher': 114,
    'Machine Learning Researcher':  'combined_114_144',
    'Security Analyst':             113,
    'Embedded Software Engineer':   182,
    'Robotics Engineer':            185,
    'Ethical Hacker':               133,
    'Computer Vision Engineer':     114,
    'DevOps Engineer':              124,
    'IoT Developer':                'combined_124_182',
    'NLP Engineer':                 144,
    'Blockchain Engineer':          136,
    'SEO Specialist':               88,
    'Data Privacy Specialist':      134,
    'Geospatial Analyst':           131,
    'Distributed Systems Engineer': 117,
    'Digital Forensics Specialist': 135,
    'Game Developer':               123,
    'Healthcare IT Specialist':     119,
    'VR Developer':                 123,
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
    'Quantum Computing Researcher': ['HPCC', 'RSCH'],
    'Machine Learning Researcher':  ['MLNG', 'RSCH'],
    'Security Analyst':             ['SCTY', 'VUAS'],
    'Embedded Software Engineer':   ['RESD', 'PROG'],
    'Robotics Engineer':            ['RESD', 'SINT'],
    'Ethical Hacker':               ['PENT', 'VUAS'],
    'Computer Vision Engineer':     ['MLNG', 'DATS'],
    'DevOps Engineer':              ['DEPL', 'RELM', 'CFMG'],
    'IoT Developer':                ['RESD', 'NTDS'],
    'NLP Engineer':                 ['MLNG', 'DATS'],
    'Blockchain Engineer':          ['PROG', 'SCTY'],
    'SEO Specialist':               ['DIGM', 'MRCH'],
    'Data Privacy Specialist':      ['PEDP', 'INAS'],
    'Geospatial Analyst':           ['DAAN', 'SCMO'],
    'Distributed Systems Engineer': ['NTDS', 'DESN'],
    'Digital Forensics Specialist': ['DGFS', 'CRIM'],
    'Game Developer':               ['ADEV', 'PROG'],
    'Healthcare IT Specialist':     ['DBAD', 'SCTY'],
    'VR Developer':                 ['ADEV', 'PROG'],
}

# ============================================================
#   Step 1 — Load datasets
# ============================================================

students = pd.read_csv(os.path.join(DATA, 'cs_students_cleaned.csv'))
courses  = pd.read_csv(os.path.join(DATA, 'Coursera_cleaned.csv'))
onet     = pd.read_excel(os.path.join(DATA, 'onet_occupation_data.xlsx'))
sfia     = pd.read_csv(os.path.join(DATA, 'sfia_standard.csv'), encoding='latin1')

# Normalise course rating: replace non-numeric with median, scale to [0, 1]
courses['Course Rating'] = pd.to_numeric(courses['Course Rating'], errors='coerce')
median_rating            = courses['Course Rating'].median()
courses['Course Rating'] = courses['Course Rating'].fillna(median_rating)
courses['rating_norm']   = courses['Course Rating'] / 5.0

print('=' * 65)
print('PIPELINE 7 — O*NET + SFIA → TF-IDF (Unfiltered)')
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
#   Step 3 — Build course corpus
#            Course Name + Description only
#            Skills tags excluded for evaluation independence
# ============================================================

course_texts = (
    courses['Course Name'].fillna('') + ' ' +
    courses['Course Description'].fillna('')
).tolist()

print(f'\nCourse corpus built: {len(course_texts)} documents')

# ============================================================
#   Step 4 — Build career queries
#            One query per unique career
#            Fit TF-IDF once per career query
# ============================================================

def recommend_tfidf(query, course_corpus, courses_df, top_n=TOP_N):
    """
    Fits TF-IDF on [query] + full course corpus.
    Returns top_n courses ranked by cosine similarity.
    """
    corpus = [query] + course_corpus
    vec    = TfidfVectorizer(stop_words='english', max_features=MAX_FEATS)
    tfidf  = vec.fit_transform(corpus)
    scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
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
#   Step 5 — Run for all students
#            TF-IDF is computed once per unique career
#            to avoid redundant computation
# ============================================================

all_recs       = []
skipped        = []
career_cache   = {}   # cache results per career

print('\nGenerating recommendations...')

for _, student in students.iterrows():
    career = student['Future Career']

    if career not in career_cache:
        onet_desc = get_onet_description(career, onet, CAREER_ONET_ROW)
        sfia_desc = get_sfia_description(CAREER_SFIA_CODES.get(career, []), sfia)

        if not onet_desc and not sfia_desc:
            skipped.append(career)
            career_cache[career] = None
            continue

        query               = f"{career} {onet_desc} {sfia_desc}"
        ranked              = rerank(recommend_tfidf(query, course_texts, courses))
        career_cache[career] = ranked
        print(f'  Computed: {career}')

    ranked = career_cache.get(career)
    if not ranked:
        continue

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
#   Step 6 — Save results
# ============================================================

os.makedirs(OUT, exist_ok=True)
out_path = os.path.join(OUT, 'p7_recommendations.csv')
results_df = pd.DataFrame(all_recs)
results_df.to_csv(out_path, index=False)

print(f'\nTotal recommendations : {len(results_df)}')
print(f'Students covered      : {results_df["student_id"].nunique()}')
print(f'Careers covered       : {results_df["future_career"].nunique()}')
if skipped:
    print(f'Skipped               : {set(skipped)}')
print(f'\nSaved → {out_path}')