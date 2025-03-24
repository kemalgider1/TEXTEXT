
import pandas as pd
from collections import defaultdict
import re

# Load data
brand_dict_df = pd.read_csv('BRAND_DICTIONARY.csv')
extracted_text_df = pd.read_csv('extracted_text.csv')

# Preprocess brand dictionary
brand_dict_df['TEXT'] = brand_dict_df['TEXT'].str.upper()
brand_dict_df['TEXT'] = brand_dict_df['TEXT'].str.replace(r'[^A-Z0-9 ]', '', regex=True)

# Separate brand families and differentiators
brand_families = brand_dict_df[brand_dict_df['TYPE'] == 'BRAND_FAMILY_NAME']['TEXT'].unique()
differentiators = brand_dict_df[brand_dict_df['TYPE'] != 'BRAND_FAMILY_NAME']['TEXT'].unique()

# Preprocess extracted text
extracted_text_df['RAW_TEXT'] = extracted_text_df['RAW_TEXT'].str.upper()
extracted_text_df['RAW_TEXT'] = extracted_text_df['RAW_TEXT'].str.replace(r'[^A-Z0-9 ]', '', regex=True)

# Match logic
image_matches = defaultdict(lambda: {'brands': [], 'differentiators': [], 'full_matches': [], 'raw_text': '', 'score': 0})

for idx, row in extracted_text_df.iterrows():
    image_id = row['IMAGE_ID']
    raw_text = row['RAW_TEXT']
    tokens = set(raw_text.split())

    matched_brands = [bf for bf in brand_families if bf in raw_text]
    matched_diffs = [diff for diff in differentiators if diff in tokens]

    full_matches = []
    for brand in matched_brands:
        for diff in matched_diffs:
            full_name = f"{brand} {diff}"
            if full_name in raw_text:
                full_matches.append(full_name)

    # Confidence score: weighted sum
    score = len(matched_brands)*2 + len(matched_diffs) + len(full_matches)*3

    image_matches[image_id]['brands'] = matched_brands
    image_matches[image_id]['differentiators'] = matched_diffs
    image_matches[image_id]['full_matches'] = full_matches
    image_matches[image_id]['raw_text'] = raw_text
    image_matches[image_id]['score'] = score

# Convert to DataFrame
results = []
for image_id, matches in image_matches.items():
    results.append({
        'IMAGE_ID': image_id,
        'RAW_TEXT': matches['raw_text'],
        'BRAND_FAMILIES_FOUND': ', '.join(matches['brands']),
        'DIFFERENTIATORS_FOUND': ', '.join(matches['differentiators']),
        'FULL_PRODUCT_MATCHES': ', '.join(matches['full_matches']),
        'CONFIDENCE_SCORE': matches['score']
    })

matches_df = pd.DataFrame(results)
matches_df.to_excel('matched_products_per_image11.xlsx', index=False)
