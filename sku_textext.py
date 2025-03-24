import boto3
import os
import csv
import pandas as pd
import re
from collections import defaultdict
from difflib import SequenceMatcher

def load_brand_dictionary(csv_path):
    """
    Load the brand dictionary from CSV file into a dictionary structure.
    The structure is optimized for brand matching with cleaning and normalization.
    """
    brand_dict = defaultdict(list)
    
    # Function to clean and normalize brand text
    def clean_text(text):
        return text.strip()
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Get header
        
        # Check if we have the expected CSV format
        if len(header) >= 2 and header[0].upper() == 'TEXT' and header[1].upper() == 'TYPE':
            for row in reader:
                if len(row) >= 2:
                    text, brand_type = row
                    brand_dict[brand_type].append(clean_text(text))
        else:
            # Try with comma separator if header doesn't match
            print("CSV header not in expected format. Trying alternative parsing...")
            # Go back to beginning of file
            f.seek(0)
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    text = parts[0]
                    brand_type = parts[1]
                    if text != 'TEXT' and brand_type != 'TYPE':  # Skip header
                        brand_dict[brand_type].append(clean_text(text))
    
    # Print summary of loaded data
    total_brands = sum(len(brands) for brands in brand_dict.values())
    print(f"Loaded {total_brands} brands across {len(brand_dict)} categories from {csv_path}")
    for brand_type, brands in brand_dict.items():
        print(f"  - {brand_type}: {len(brands)} brands")
    
    return brand_dict

def detect_text(photo_path, bucket=None):
    """
    Detect text in an image using Amazon Rekognition.
    For local images, set bucket to None.
    For S3 images, provide bucket name and photo path in S3.
    Returns all detected texts with confidence levels.
    """
    session = boto3.Session(profile_name='default')
    client = session.client('rekognition')
    
    try:
        if bucket:
            # S3 image
            response = client.detect_text(
                Image={'S3Object': {'Bucket': bucket, 'Name': photo_path}}
            )
        else:
            # Local image
            with open(photo_path, 'rb') as image:
                response = client.detect_text(
                    Image={'Bytes': image.read()}
                )
        
        # Get both LINE and WORD detections
        line_texts = []
        word_texts = []
        for text_detection in response['TextDetections']:
            detection_type = text_detection['Type']
            text = text_detection['DetectedText']
            confidence = text_detection['Confidence']
            
            if detection_type == 'LINE':
                line_texts.append((text, confidence))
            elif detection_type == 'WORD':
                word_texts.append((text, confidence))
        
        # Debug output
        print(f"Detected {len(line_texts)} lines and {len(word_texts)} words in {photo_path}")
        for line, conf in line_texts:
            print(f"  LINE: {line} (Confidence: {conf:.2f}%)")
        
        # Return mainly the line texts, but include individual words if they might be important
        # for brand detection (often brand names can be a single word)
        detected_texts = [text for text, _ in line_texts]
        
        # Add high-confidence words that might be missed in line detection
        for word, conf in word_texts:
            if conf > 90 and word not in detected_texts and len(word) > 2:
                detected_texts.append(word)
        
        return detected_texts
        
    except Exception as e:
        print(f"Error in detect_text for {photo_path}: {str(e)}")
        # Return an empty list rather than failing completely
        return []

def match_text_with_brands(detected_texts, brand_dict):
    """
    Advanced matching algorithm to identify brands in detected text.
    Uses multiple techniques including exact matching, fuzzy matching,
    and context-aware matching to improve accuracy.
    """
    matches = defaultdict(list)
    
    # Helper function to clean text for matching
    def clean_text(text):
        # Convert to lowercase and remove special characters
        cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
        # Replace multiple spaces with a single space
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    
    # Prepare cleaned versions of detected texts
    cleaned_texts = [clean_text(text) for text in detected_texts]
    full_cleaned_text = " ".join(cleaned_texts)
    
    # Create a list of individual words from all detected texts
    all_words = set()
    for text in cleaned_texts:
        all_words.update(text.split())
    
    # Function to calculate similarity between two strings
    def similarity_ratio(a, b):
        return SequenceMatcher(None, a, b).ratio()
    
    # Function to calculate more comprehensive match score
    def calculate_match_score(brand, threshold=0.75):
        brand_clean = clean_text(brand)
        brand_words = brand_clean.split()
        
        # Case 1: Direct exact match in any of the detected texts
        for text in cleaned_texts:
            if brand_clean in text:
                # Perfect match
                return 1.0
        
        # Case 2: Check if all words in the brand appear in the detected text
        # This helps with brands that might be split across multiple lines
        if all(word in full_cleaned_text for word in brand_words):
            # All words appear but not necessarily together
            return 0.9
        
        # Case 3: Check for fuzzy matches (useful for OCR errors)
        best_fuzzy_score = 0
        for text in cleaned_texts:
            # Try sliding window approach for multi-word brands
            if len(brand_words) > 1:
                text_words = text.split()
                for i in range(len(text_words) - len(brand_words) + 1):
                    window = " ".join(text_words[i:i+len(brand_words)])
                    score = similarity_ratio(window, brand_clean)
                    best_fuzzy_score = max(best_fuzzy_score, score)
            else:
                # For single word brands, check similarity with each word
                for word in text.split():
                    score = similarity_ratio(word, brand_clean)
                    best_fuzzy_score = max(best_fuzzy_score, score)
        
        if best_fuzzy_score >= threshold:
            return best_fuzzy_score
        
        # Case 4: Check if a significant portion of brand words appear
        if len(brand_words) > 1:
            matched_words = sum(1 for word in brand_words if word in all_words)
            word_match_ratio = matched_words / len(brand_words)
            if word_match_ratio >= 0.6:  # At least 60% of words match
                return 0.7 * word_match_ratio
        
        return 0
    
    # Check for matches in each brand type
    for brand_type, brands in brand_dict.items():
        for brand in brands:
            score = calculate_match_score(brand)
            if score > 0.5:  # Only include reasonably confident matches
                matches[brand_type].append((brand, score))
    
    # Sort matches by score within each type

def process_images_directory(images_dir, brand_dict_path, output_csv):
    """
    Process all images in a directory and create a CSV report.
    """
    # Load brand dictionary
    brand_dict = load_brand_dictionary(brand_dict_path)
    
    # Prepare results table
    results = []
    
    # Get list of image files
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        image_id = os.path.splitext(image_file)[0]
        
        # Detect text in image
        try:
            detected_texts = detect_text(image_path)
            raw_text = " | ".join(detected_texts)
            
            # Match with brands
            matches = match_text_with_brands(detected_texts, brand_dict)
            
            # Flatten matches for each type, extracting just the brand names (not scores)
            brand_family_matches = [match[0] for match in matches.get('BRAND_FAMILY_NAME', [])]
            differentiator_matches = [match[0] for match in matches.get('BRAND_DIFFERENTIATOR_NAME', [])]
            combined_matches = [match[0] for match in matches.get('COMBINED', [])]
            
            # Prioritize in this order: Combined, Family, Differentiator
            # This assumes combined entries are more specific and should take precedence
            prioritized_matches = combined_matches + brand_family_matches + differentiator_matches
            
            # Remove duplicates while preserving order
            unique_matches = []
            for match in prioritized_matches:
                if match not in unique_matches:
                    unique_matches.append(match)
            
            # Create row with image_id, raw_text, and up to 10 predicted SKUs
            row = [image_id, raw_text] + unique_matches[:10] + [''] * (10 - len(unique_matches[:10]))
            results.append(row)
            
            print(f"Processed {image_file}: Found {len(unique_matches)} unique matches")
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            row = [image_id, f"ERROR: {str(e)}"] + [''] * 10
            results.append(row)
    
    # Create DataFrame and save to CSV
    headers = ['IMAGE_ID', 'RAW_TEXT'] + [f'PREDICTED_SKUs{i+1}' for i in range(10)]
    df = pd.DataFrame(results, columns=headers)
    df.to_csv(output_csv, index=False)
    
    print(f"Results saved to {output_csv}")
    return df

def main():
    """
    Main function to run the image text analysis process.
    """
    # Set your paths
    images_dir = 'images'
    brand_dict_path = 'BRAND_DICTIONARY.csv'
    output_csv = 'image_text_analysis_results.csv'
    
    # Check if directories and files exist
    if not os.path.exists(images_dir):
        print(f"Warning: Images directory '{images_dir}' not found. Creating it...")
        os.makedirs(images_dir)
        print(f"Please place your images in the '{images_dir}' directory and run the script again.")
        return
    
    if not os.path.exists(brand_dict_path):
        print(f"Error: Brand dictionary file '{brand_dict_path}' not found.")
        print("Make sure 'BRAND_DICTIONARY.csv' is in the same directory as this script.")
        return
    
    # Check if there are any images to process
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
    if not image_files:
        print(f"No image files found in '{images_dir}'. Please add images and run again.")
        return
    
    print(f"Starting analysis of {len(image_files)} images...")
    
    # Process images and generate report
    results_df = process_images_directory(images_dir, brand_dict_path, output_csv)
    
    # Display summary
    print(f"\nProcessed {len(results_df)} images")
    matched_images = results_df[results_df['PREDICTED_SKUs1'] != ''].shape[0]
    print(f"Images with at least one match: {matched_images}")
    print(f"Images with no matches: {len(results_df) - matched_images}")
    
    # Display top matches by frequency
    print("\nTop Brand Matches:")
    all_matches = []
    for i in range(1, 11):
        col = f'PREDICTED_SKUs{i}'
        if col in results_df.columns:
            all_matches.extend([m for m in results_df[col].tolist() if m])
    
    if all_matches:
        from collections import Counter
        top_matches = Counter(all_matches).most_common(10)
        for brand, count in top_matches:
            print(f"  {brand}: {count} occurrences")
    else:
        print("  No matches found in any images")
    
    print(f"\nResults saved to {output_csv}")

if __name__ == "__main__":
    print("Text-in-Image Brand Matching Tool")
    print("--------------------------------")
    main()
