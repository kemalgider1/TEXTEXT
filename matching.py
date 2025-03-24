import pandas as pd
import csv
from collections import defaultdict
from tqdm import tqdm
from difflib import get_close_matches  # For fuzzy matching
import re

def match_product(raw_text, brand_dict, verbose=False):
    """Enhanced matching with fuzzy matching for OCR errors"""
    matches = []
    raw_text = raw_text.upper()
    words = [w.strip() for w in re.split(r'[|\s]+', raw_text) if len(w.strip()) > 1]

    # Fuzzy match for brands in text
    if 'BRAND_FAMILY_NAME' in brand_dict:
        brand_matches = {}
        brands_upper = [b.upper() for b in brand_dict['BRAND_FAMILY_NAME']]

        # Find potential brands with fuzzy matching
        for word in words:
            if len(word) >= 4:  # Only consider words of sufficient length
                close_matches = get_close_matches(word, brands_upper, n=3, cutoff=0.7)
                for match_idx, brand_upper in enumerate(close_matches):
                    brand_idx = brands_upper.index(brand_upper)
                    brand = brand_dict['BRAND_FAMILY_NAME'][brand_idx]
                    confidence = 90 - (match_idx * 10)  # Higher confidence for better matches

                    if brand not in brand_matches or confidence > brand_matches[brand]['confidence']:
                        brand_matches[brand] = {'confidence': confidence, 'differentiators': []}

        # If we found potential brands, look for differentiators
        for brand in brand_matches:
            if 'BRAND_DIFFERENTIATOR_NAME' in brand_dict:
                diffs_upper = [d.upper() for d in brand_dict['BRAND_DIFFERENTIATOR_NAME']]

                for word in words:
                    if len(word) >= 3:  # Only consider words of sufficient length
                        close_diff_matches = get_close_matches(word, diffs_upper, n=3, cutoff=0.75)
                        for diff_upper in close_diff_matches:
                            diff_idx = diffs_upper.index(diff_upper)
                            diff = brand_dict['BRAND_DIFFERENTIATOR_NAME'][diff_idx]
                            combined = f"{brand} {diff}"

                            # Check if this is a valid product
                            if 'COMBINED' in brand_dict and combined in brand_dict['COMBINED']:
                                matches.append((combined, 85))
                            else:
                                brand_matches[brand]['differentiators'].append(diff)

        # Create matches from brand+differentiator combinations
        for brand, info in brand_matches.items():
            if not matches:  # Only use if we don't have combined matches
                if info['differentiators']:
                    # Find most frequently occurring differentiator
                    matches.append((f"{brand} {info['differentiators'][0]}", 75))
                else:
                    matches.append((brand, 60))

    # Sort by confidence
    matches = sorted(matches, key=lambda x: x[1], reverse=True)

    # Remove duplicates while preserving order
    unique_matches = []
    seen = set()
    for match, _ in matches:
        if match not in seen:
            unique_matches.append(match)
            seen.add(match)

    return unique_matches