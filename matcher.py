import pandas as pd
import numpy as np
import re
from collections import defaultdict
import difflib
from typing import List, Dict, Tuple, Set, Optional


class ProductMatcher:
    """
    Enhanced product matching system for PMI products from image text.
    Implements improved tokenization, proximity-based matching, and confidence scoring.
    """

    def __init__(self, brand_dict_path: str, min_confidence: float = 0.6, verbose: bool = False):
        """
        Initialize the product matcher with brand dictionary and configuration.

        Args:
            brand_dict_path: Path to the brand dictionary CSV
            min_confidence: Minimum confidence threshold for matches (0-1)
            verbose: Whether to print additional information
        """
        self.min_confidence = min_confidence
        self.verbose = verbose
        self.brand_dict_df = self._load_brand_dictionary(brand_dict_path)

        # Create efficient lookup structures
        self.brand_families = set(
            self.brand_dict_df[self.brand_dict_df['TYPE'] == 'BRAND_FAMILY_NAME']['TEXT'].str.upper())
        self.differentiators = set(
            self.brand_dict_df[self.brand_dict_df['TYPE'] == 'BRAND_DIFFERENTIATOR_NAME']['TEXT'].str.upper())
        self.combined_products = set(self.brand_dict_df[self.brand_dict_df['TYPE'] == 'COMBINED']['TEXT'].str.upper())

        if self.verbose:
            print(f"Loaded {len(self.brand_families)} brand families")
            print(f"Loaded {len(self.differentiators)} differentiators")
            print(f"Loaded {len(self.combined_products)} combined products")

        # Create mappings for fuzzy matching
        self._create_token_mappings()

    def _load_brand_dictionary(self, path: str) -> pd.DataFrame:
        """Load and preprocess the brand dictionary."""
        df = pd.read_csv(path)
        df['TEXT'] = df['TEXT'].str.upper()
        return df

    def _create_token_mappings(self) -> None:
        """Create mappings to support fuzzy matching and error correction."""
        # Create normalized forms of brand names and differentiators
        self.normalized_brands = {self._normalize_text(brand): brand for brand in self.brand_families}
        self.normalized_differentiators = {self._normalize_text(diff): diff for diff in self.differentiators}
        self.normalized_combined = {self._normalize_text(prod): prod for prod in self.combined_products}

        # Create prefix lookup for partial matches (first 3-4 chars)
        self.brand_prefixes = defaultdict(list)
        for brand in self.brand_families:
            if len(brand) >= 3:
                self.brand_prefixes[brand[:3]].append(brand)
                if len(brand) >= 4:
                    self.brand_prefixes[brand[:4]].append(brand)

    def _normalize_text(self, text: str) -> str:
        """Normalize text by removing spaces and special characters."""
        return re.sub(r'[^A-Z0-9]', '', text.upper())

    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words with proper handling of noise and OCR errors.
        Returns a list of cleaned tokens.
        """
        # Initial cleaning
        text = text.upper()

        # Remove common OCR artifacts and normalize spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        # Tokenize
        tokens = text.split()

        # Remove very short tokens (likely OCR errors)
        tokens = [t for t in tokens if len(t) > 1]

        return tokens

    def _find_brand_candidates(self, tokens: List[str], text: str) -> List[Tuple[str, float, int]]:
        """
        Find potential brand family name matches in the text.
        Returns a list of (brand, confidence, position) tuples.
        """
        candidates = []

        # Method 1: Direct token matching
        for i, token in enumerate(tokens):
            if token in self.brand_families:
                # Direct match
                candidates.append((token, 1.0, i))
            elif len(token) >= 3:
                # Check for prefix matches (handle truncated brand names)
                prefix = token[:3]
                if prefix in self.brand_prefixes:
                    for brand in self.brand_prefixes[prefix]:
                        similarity = difflib.SequenceMatcher(None, token, brand).ratio()
                        if similarity > 0.7:  # Threshold for fuzzy prefix match
                            candidates.append((brand, similarity, i))

        # Method 2: Substring matching for major brands in the full text
        # This helps with cases where tokenization fails or OCR merges words
        for brand in self.brand_families:
            if len(brand) > 4:  # Only check substantial brand names to avoid false positives
                if brand in text:
                    # Find position in token list
                    pos = -1
                    for i, token in enumerate(tokens):
                        if token in brand or brand in token:
                            pos = i
                            break

                    if pos >= 0 and not any(b[0] == brand for b in candidates):
                        candidates.append((brand, 0.9, pos))  # Slightly lower confidence for substring match

        # Sort by confidence descending
        return sorted(candidates, key=lambda x: x[1], reverse=True)

    def _find_differentiator_candidates(self, tokens: List[str], text: str) -> List[Tuple[str, float, int]]:
        """
        Find potential differentiator matches in the text.
        Returns a list of (differentiator, confidence, position) tuples.
        """
        candidates = []

        # Method 1: Direct token matching
        for i, token in enumerate(tokens):
            if token in self.differentiators:
                candidates.append((token, 1.0, i))
            else:
                # Fuzzy matching for differentiators
                for diff in self.differentiators:
                    if len(diff) > 3 and len(token) > 3:
                        similarity = difflib.SequenceMatcher(None, token, diff).ratio()
                        if similarity > 0.8:  # Higher threshold for differentiators
                            candidates.append((diff, similarity, i))

        # Method 2: Look for multi-word differentiators
        for diff in self.differentiators:
            if ' ' in diff:
                diff_parts = diff.split()
                if all(part in tokens for part in diff_parts):
                    # Find the position (use the first token's position)
                    pos = -1
                    for i, token in enumerate(tokens):
                        if token == diff_parts[0]:
                            pos = i
                            break

                    if pos >= 0:
                        candidates.append((diff, 0.95, pos))  # High confidence for multi-word match

        # Sort by confidence descending
        return sorted(candidates, key=lambda x: x[1], reverse=True)

    def _find_full_product_matches(self,
                                   tokens: List[str],
                                   text: str,
                                   brand_candidates: List[Tuple[str, float, int]],
                                   diff_candidates: List[Tuple[str, float, int]]) -> List[Tuple[str, float]]:
        """
        Find full product matches by combining brands and differentiators.
        Uses proximity analysis and dictionary validation.
        Returns a list of (product_name, confidence) tuples.
        """
        matches = []

        # Method 1: Check for combined products directly in text
        for combined in self.combined_products:
            # Direct full match in original text (handles spacing variation)
            normalized_text = self._normalize_text(text)
            normalized_combined = self._normalize_text(combined)

            if normalized_combined in normalized_text:
                matches.append((combined, 0.95))
                continue

            # Check for token sequence match
            combined_tokens = combined.split()
            if len(combined_tokens) > 1:
                # Check if all tokens appear in sequence
                combined_found = True
                token_positions = []

                for c_token in combined_tokens:
                    found = False
                    for i, token in enumerate(tokens):
                        if c_token in token or token in c_token:
                            token_positions.append(i)
                            found = True
                            break

                    if not found:
                        combined_found = False
                        break

                # Check if tokens are sequential or close (allowing for 1 token gap)
                if combined_found:
                    token_positions.sort()
                    is_sequential = all(token_positions[i + 1] - token_positions[i] <= 2
                                        for i in range(len(token_positions) - 1))

                    if is_sequential:
                        matches.append((combined, 0.9))

        # Method 2: Proximity-based brand + differentiator matching
        for brand, brand_conf, brand_pos in brand_candidates:
            for diff, diff_conf, diff_pos in diff_candidates:
                # Skip if already matched as a combined product
                combined = f"{brand} {diff}"
                if any(m[0] == combined for m in matches):
                    continue

                # CRITICAL CHECK: Verify combined product exists in dictionary
                if combined in self.combined_products:
                    # Calculate proximity score (higher when closer together)
                    proximity = 1.0 / (1.0 + abs(brand_pos - diff_pos))

                    # Calculate combined confidence
                    confidence = (brand_conf * 0.5 + diff_conf * 0.3 + proximity * 0.2)

                    if confidence >= self.min_confidence:
                        matches.append((combined, confidence))

        # Remove duplicates and sort by confidence
        unique_matches = []
        seen = set()

        # Validate all matches against the COMBINED list from dictionary
        for product, conf in sorted(matches, key=lambda x: x[1], reverse=True):
            if product not in seen and product in self.combined_products:
                unique_matches.append((product, conf))
                seen.add(product)

        return unique_matches

    def _calculate_overall_confidence(self,
                                      full_matches: List[Tuple[str, float]],
                                      brand_matches: List[Tuple[str, float, int]],
                                      diff_matches: List[Tuple[str, float, int]]) -> float:
        """Calculate an overall confidence score for the matches."""
        if not full_matches and not brand_matches:
            return 0.0

        # Calculate factors for confidence
        full_match_score = max([conf for _, conf in full_matches]) if full_matches else 0
        brand_match_score = max([conf for _, conf, _ in brand_matches]) if brand_matches else 0
        diff_match_score = max([conf for _, conf, _ in diff_matches]) if diff_matches else 0
        match_count_factor = min(1.0, (len(full_matches) + len(brand_matches) * 0.5) / 3)

        # Weight the factors (full matches are most important)
        confidence = (full_match_score * 0.6 +
                      brand_match_score * 0.25 +
                      diff_match_score * 0.05 +
                      match_count_factor * 0.1)

        return min(1.0, confidence)

    def match_products(self, extracted_text_df) -> pd.DataFrame:
        """
        Process a dataframe of extracted text and identify product matches.

        Args:
            extracted_text_df: DataFrame with IMAGE_ID and RAW_TEXT columns
                              OR an iterator of (index, row) tuples (for tqdm support)

        Returns:
            DataFrame with match results per image
        """
        results = []
        total_images = 0

        # Track statistics for progress updates
        matches_found = 0

        # Check if extracted_text_df is an iterator or DataFrame
        is_iterator = not isinstance(extracted_text_df, pd.DataFrame)

        # Process each image
        for item in extracted_text_df:
            # Handle both DataFrame and iterator formats
            if is_iterator:
                idx, row = item
            else:
                _, row = item

            image_id = row['IMAGE_ID']
            raw_text = row['RAW_TEXT']

            # Preprocess and tokenize
            tokens = self._tokenize_text(raw_text)

            # Find brand and differentiator candidates
            brand_candidates = self._find_brand_candidates(tokens, raw_text)
            diff_candidates = self._find_differentiator_candidates(tokens, raw_text)

            # Get full product matches
            full_matches = self._find_full_product_matches(tokens, raw_text, brand_candidates, diff_candidates)

            # Calculate overall confidence score
            confidence_score = self._calculate_overall_confidence(full_matches, brand_candidates, diff_candidates)
            confidence_score = round(confidence_score * 10)  # Scale to 0-10 range

            # Extract final lists
            brand_families = [b[0] for b in brand_candidates]
            differentiators = [d[0] for d in diff_candidates]
            full_product_matches = [m[0] for m in full_matches]

            # Remove duplicates while preserving order
            brand_families = list(dict.fromkeys(brand_families))
            differentiators = list(dict.fromkeys(differentiators))
            full_product_matches = list(dict.fromkeys(full_product_matches))

            # Update match counter for reporting
            if full_product_matches:
                matches_found += 1

                # Print occasional updates about interesting matches (every 100 images)
                if self.verbose and total_images % 100 == 0 and full_product_matches:
                    print(f"\n  Found match: {full_product_matches[0]} (Confidence: {confidence_score}/10)")

            # Build result row
            result = {
                'IMAGE_ID': image_id,
                'RAW_TEXT': raw_text,
                'BRAND_FAMILIES_FOUND': ', '.join(brand_families),
                'DIFFERENTIATORS_FOUND': ', '.join(differentiators),
                'FULL_PRODUCT_MATCHES': ', '.join(full_product_matches),
                'CONFIDENCE_SCORE': confidence_score
            }

            results.append(result)
            total_images += 1

        # Print summary if verbose
        if self.verbose:
            print(
                f"\nMatching complete: Found {matches_found} full product matches ({matches_found / total_images * 100:.1f}%)")

        # Convert to DataFrame
        return pd.DataFrame(results)


def main():
    """Main entry point for the enhanced product matching system."""
    import time
    start_time = time.time()

    # Load input data
    print("=" * 80)
    print("ENHANCED PRODUCT MATCHER")
    print("=" * 80)
    print("\nInitializing system...")

    brand_dict_path = 'BRAND_DICTIONARY.csv'
    extracted_text_path = 'extracted_text.csv'
    output_path = 'product_matches.csv'

    print(f"• Loading brand dictionary from {brand_dict_path}")
    try:
        brand_df = pd.read_csv(brand_dict_path)
        num_brands = brand_df[brand_df['TYPE'] == 'BRAND_FAMILY_NAME'].shape[0]
        num_diffs = brand_df[brand_df['TYPE'] == 'BRAND_DIFFERENTIATOR_NAME'].shape[0]
        num_combined = brand_df[brand_df['TYPE'] == 'COMBINED'].shape[0]
        print(f"  Successfully loaded {len(brand_df)} entries:")
        print(f"  - {num_brands} brand families")
        print(f"  - {num_diffs} differentiators")
        print(f"  - {num_combined} combined products")
    except Exception as e:
        print(f"  ERROR: Could not load brand dictionary - {e}")
        return

    print(f"\n• Loading input data from {extracted_text_path}")
    try:
        extracted_text_df = pd.read_csv(extracted_text_path)
        print(f"  Successfully loaded {len(extracted_text_df)} image text records")
    except Exception as e:
        print(f"  ERROR: Could not load extracted text file - {e}")
        return

    # Initialize matcher
    print("\n• Initializing product matcher...")
    matcher = ProductMatcher(brand_dict_path, verbose=True)
    print("  Matcher initialized successfully")

    # Process all images
    print(f"\n• Processing {len(extracted_text_df)} images...")

    # Add progress tracking with tqdm if available
    try:
        from tqdm import tqdm
        # Wrap our dataframe iteration with tqdm for a progress bar
        extracted_text_df = tqdm(extracted_text_df.iterrows(), total=len(extracted_text_df),
                                 desc="Matching products", unit="image")
    except ImportError:
        # tqdm not available, just use regular iteration
        extracted_text_df = extracted_text_df.iterrows()

    results_df = matcher.match_products(extracted_text_df)

    # Save results
    print(f"\n• Saving results to {output_path}...")
    results_df.to_csv(output_path, index=False)
    print(f"  Results saved successfully")

    # Print summary
    end_time = time.time()
    elapsed_time = end_time - start_time
    images_per_second = len(results_df) / elapsed_time

    full_matches = results_df[results_df['FULL_PRODUCT_MATCHES'] != ''].shape[0]
    brand_matches = results_df[results_df['BRAND_FAMILIES_FOUND'] != ''].shape[0]

    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)
    print(f"Total runtime: {elapsed_time:.2f} seconds ({images_per_second:.1f} images/second)")
    print(f"Processed {len(results_df)} images:")
    print(f"• {brand_matches} images with brand matches ({brand_matches / len(results_df) * 100:.1f}%)")
    print(f"• {full_matches} images with full product matches ({full_matches / len(results_df) * 100:.1f}%)")
    print(f"• Average confidence score: {results_df['CONFIDENCE_SCORE'].mean():.2f}/10")

    # List top 5 most common products found
    if full_matches > 0:
        print("\nTop 5 most common product matches:")
        all_products = []
        for products in results_df['FULL_PRODUCT_MATCHES'].dropna():
            if products:
                all_products.extend([p.strip() for p in products.split(',')])

        for i, (product, count) in enumerate(pd.Series(all_products).value_counts().head(5).items(), 1):
            print(f"  {i}. {product} ({count} instances)")

    print("\nDone!")


if __name__ == "__main__":
    main()