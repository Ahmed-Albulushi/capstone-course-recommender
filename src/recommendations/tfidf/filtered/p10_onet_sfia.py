# ============================================================
# Pipeline 10 — O*NET + SFIA → TF-IDF (Filtered)
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
# Domain filter: Applied post-scoring — NO fallback
#   Keep only courses tagged as:
#   computer-science, data-science, information-technology
#   Candidate pool: top 50 → filter → top 10
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
#   - results/recommendations/tfidf/filtered/p10_recommendations.csv
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
OUT  = os.path.join(BASE, 'results', 'recommendations', 'tfidf', 'filtered')

# ============================================================
#   Configuration
# ============================================================

TOP_N          = 10
TOP_CANDIDATES = 50     # fetch before domain filter
MAX_FEATS      = 20000

CS_DOMAINS = {
    'computer-science',
    'data-science',
    'information-technology',
}

KNOWN_DOMAINS = {
    'business', 'computer-science', 'data-science', 'life-sciences',
    'physical-science-and-engineering', 'social-sciences', 'arts-and-humanities',
    'information-technology', 'language-learning', 'personal-development', 'math-and-logic'
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

# Extract broad domain from Skills tag (second-to-last token)
def extract_broad_domain(skills_str):
    if not isinstance(skills_str, str) or not skills_str.strip():
        return 'unknown'
    tokens = skills_str.strip().split()
    if len(tokens) < 2:
        return 'unknown'
    candidate = tokens[-2]
    return candidate if candidate in KNOWN_DOMAINS else 'unknown'

courses['broad_domain'] = courses['Skills'].apply(extract_broad_domain)
cs_count = courses['broad_domain'].isin(CS_DOMAINS).sum()

print('=' * 65)
print('PIPELINE 10 — O*NET + SFIA → TF-IDF (Filtered)')
print('=' * 65)
print(f'Students          : {len(students)}')
print(f'Courses           : {len(courses)}')
print(f'CS-domain courses : {cs_count}')
print(f'Top N             : {TOP_N}')
print(f'Candidate pool    : {TOP_CANDIDATES}')

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
# ============================================================

course_texts = (
    courses['Course Name'].fillna('') + ' ' +
    courses['Course Description'].fillna('')
).tolist()

print(f'\nCourse corpus built: {len(course_texts)} documents')

# ============================================================
#   Step 4 — TF-IDF matching + domain filter
# ============================================================

def recommend_tfidf_filtered(query, course_corpus, courses_df,
                              top_candidates=TOP_CANDIDATES, top_n=TOP_N):
    """
    Fits TF-IDF, scores all courses, returns top_candidates.
    Then filters to CS-domain only and trims to top_n.
    Falls back to unfiltered if fewer than top_n CS courses found.
    """
    corpus  = [query] + course_corpus
    vec     = TfidfVectorizer(stop_words='english', max_features=MAX_FEATS)
    tfidf   = vec.fit_transform(corpus)
    scores  = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    top_idx = scores.argsort()[-top_candidates:][::-1]

    candidates = [
        {
            'course':       courses_df.iloc[i]['Course Name'],
            'sim_score':    round(float(scores[i]), 4),
            'rating':       courses_df.iloc[i]['Course Rating'],
            'rating_norm':  round(courses_df.iloc[i]['rating_norm'], 4),
            'level':        courses_df.iloc[i]['Difficulty Level'],
            'skills':       courses_df.iloc[i]['Skills'],
            'broad_domain': courses_df.iloc[i]['broad_domain'],
        }
        for i in top_idx
    ]

    # Filter to CS-domain only — no fallback
    # Returns however many CS courses are found (may be < top_n)
    cs_candidates = [c for c in candidates if c['broad_domain'] in CS_DOMAINS]
    return cs_candidates[:top_n]

def rerank(matches):
    for c in matches:
        c['final_score'] = round(c['sim_score'] * c['rating_norm'], 4)
    return sorted(matches, key=lambda x: x['final_score'], reverse=True)

# ============================================================
#   Step 5 — Run for all students (cached per career)
# ============================================================

all_recs     = []
skipped      = []
career_cache = {}

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

        query                = f"{career} {onet_desc} {sfia_desc}"
        candidates           = recommend_tfidf_filtered(query, course_texts, courses)
        ranked               = rerank(candidates)
        career_cache[career] = ranked

        if len(ranked) < TOP_N:
            print(f'  [INFO] {career} — only {len(ranked)} CS courses found')
        else:
            print(f'  Computed: {career}')

    ranked = career_cache.get(career)
    if not ranked:
        continue
    for rank, c in enumerate(ranked, 1):
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
            'broad_domain':      c['broad_domain'],
            'sim_score':         c['sim_score'],
            'final_score':       c['final_score'],

        })

# ============================================================
#   Step 6 — Save results with metadata header
# ============================================================

os.makedirs(OUT, exist_ok=True)
out_path   = os.path.join(OUT, 'p10_recommendations.csv')
results_df = pd.DataFrame(all_recs)

# Write metadata header then data
with open(out_path, 'w') as f:
    f.write('# Pipeline      : P10 — O*NET + SFIA → TF-IDF (Filtered)\n')
    f.write('# Query         : Career title + O*NET description + SFIA descriptions\n')
    f.write('# Retrieval     : TF-IDF cosine similarity\n')
    f.write('# Domain filter : computer-science, data-science, information-technology\n')
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
print(f'CS-domain rate        : {results_df["broad_domain"].isin(CS_DOMAINS).mean()*100:.1f}%')
if skipped:
    print(f'Skipped               : {set(skipped)}')
print(f'\nSaved → {out_path}')