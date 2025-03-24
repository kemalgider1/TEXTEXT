import concurrent.futures
from tqdm import tqdm  # For simple progress bar
import os
import boto3
import csv
import pandas as pd
from collections import defaultdict
def load_brand_dictionary(csv_path, verbose=False):
    """Optimized brand dictionary loader with optional output"""
    brand_dict = defaultdict(list)

    def clean_text(text):
        return text.strip()

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        if len(header) >= 2 and header[0].upper() == 'TEXT' and header[1].upper() == 'TYPE':
            for row in reader:
                if len(row) >= 2:
                    text, brand_type = row
                    brand_dict[brand_type].append(clean_text(text))
        else:
            f.seek(0)
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    text, brand_type = parts[0], parts[1]
                    if text != 'TEXT' and brand_type != 'TYPE':
                        brand_dict[brand_type].append(clean_text(text))

    if verbose:
        # Only print if verbose mode enabled
        total_brands = sum(len(brands) for brands in brand_dict.values())
        print(f"Loaded {total_brands} brands across {len(brand_dict)} categories from {csv_path}")

    return brand_dict
def detect_text(photo_path, bucket=None, verbose=False):
    """Optimized text detection with minimal output"""
    session = boto3.Session(profile_name='default')
    client = session.client('rekognition')

    try:
        if bucket:
            response = client.detect_text(
                Image={'S3Object': {'Bucket': bucket, 'Name': photo_path}}
            )
        else:
            with open(photo_path, 'rb') as image:
                response = client.detect_text(
                    Image={'Bytes': image.read()}
                )

        if not response or 'TextDetections' not in response:
            if verbose:
                print(f"Warning: Invalid response for {photo_path}")
            return []

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

        # Debug output only if verbose is True
        if verbose:
            print(f"Detected {len(line_texts)} lines in {photo_path}")

        detected_texts = [text for text, _ in line_texts]
        for word, conf in word_texts:
            if conf > 90 and word not in detected_texts and len(word) > 2:
                detected_texts.append(word)

        return detected_texts

    except Exception as e:
        if verbose:
            print(f"Error in detect_text for {photo_path}: {str(e)}")
        return []
def process_single_image(image_file, images_dir, brand_dict, verbose=False):
    """Process a single image - for parallel execution"""
    image_path = os.path.join(images_dir, image_file)
    image_id = os.path.splitext(image_file)[0]

    try:
        detected_texts = detect_text(image_path, verbose=verbose)
        raw_text = " | ".join(detected_texts)

        # Match with brands
        matches = match_text_with_brands(detected_texts, brand_dict)

        # Extract matches by type
        brand_family_matches = [match[0] for match in matches.get('BRAND_FAMILY_NAME', [])]
        differentiator_matches = [match[0] for match in matches.get('BRAND_DIFFERENTIATOR_NAME', [])]
        combined_matches = [match[0] for match in matches.get('COMBINED', [])]

        prioritized_matches = combined_matches + brand_family_matches + differentiator_matches

        # Remove duplicates while preserving order
        unique_matches = []
        for match in prioritized_matches:
            if match not in unique_matches:
                unique_matches.append(match)

        # Create row for results
        row = [image_id, raw_text] + unique_matches[:10] + [''] * (10 - len(unique_matches[:10]))

        if verbose:
            print(f"Processed {image_file}: Found {len(unique_matches)} matches")

        return row
    except Exception as e:
        if verbose:
            print(f"Error processing {image_file}: {str(e)}")
        return [image_id, f"ERROR: {str(e)}"] + [''] * 10
def process_images_directory(images_dir, brand_dict_path, output_csv, verbose=False):
    """Process all images in a directory using parallel processing"""
    # Load brand dictionary
    brand_dict = load_brand_dictionary(brand_dict_path, verbose=verbose)

    # Get list of image files
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]

    results = []

    # Process images in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Create futures for each image
        future_to_image = {
            executor.submit(process_single_image, image_file, images_dir, brand_dict, verbose):
            image_file for image_file in image_files
        }

        # Process results as they complete with a progress bar
        for future in tqdm(concurrent.futures.as_completed(future_to_image),
                           total=len(image_files),
                           desc="Processing images",
                           disable=not verbose):
            results.append(future.result())

    # Create DataFrame and save to CSV
    headers = ['IMAGE_ID', 'RAW_TEXT'] + [f'PREDICTED_SKUs{i+1}' for i in range(10)]
    df = pd.DataFrame(results, columns=headers)
    df.to_csv(output_csv, index=False)

    if verbose:
        print(f"Results saved to {output_csv}")
    return df
def match_text_with_brands(detected_texts, brand_dict):
    """
    Match detected text strings with brand names from the dictionary.
    Returns a dictionary of matches by brand type, each with similarity scores.
    """
    matches = defaultdict(list)

    # Normalize detected texts for better matching
    normalized_texts = [text.strip().upper() for text in detected_texts]

    # For each brand type in the dictionary
    for brand_type, brands in brand_dict.items():
        # For each brand in this type
        for brand in brands:
            # Normalize brand name
            norm_brand = brand.strip().upper()

            # Simple exact matching
            for text in normalized_texts:
                if norm_brand in text or text in norm_brand:
                    # Calculate a simple score based on length ratio
                    score = min(len(text), len(norm_brand)) / max(len(text), len(norm_brand)) * 100
                    matches[brand_type].append((brand, score))
                    break

    # Sort matches by score within each type
    for brand_type in matches:
        matches[brand_type] = sorted(matches[brand_type], key=lambda x: x[1], reverse=True)

    return matches
def main(verbose=False):
    """Main function with optional verbosity"""
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
        return

    # Check if there are any images to process
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
    if not image_files:
        print(f"No image files found in '{images_dir}'.")
        return

    if verbose:
        print(f"Starting analysis of {len(image_files)} images...")

    # Process images and generate report
    results_df = process_images_directory(images_dir, brand_dict_path, output_csv, verbose=verbose)

    if verbose:
        # Display summary only in verbose mode
        print(f"\nProcessed {len(results_df)} images")
        matched_images = results_df[results_df['PREDICTED_SKUs1'] != ''].shape[0]
        print(f"Images with matches: {matched_images}")
        print(f"Images with no matches: {len(results_df) - matched_images}")
    else:
        print(f"Analysis complete. Results saved to {output_csv}")
if __name__ == "__main__":
    # Set verbose=False for minimal console output
    main(verbose=False)