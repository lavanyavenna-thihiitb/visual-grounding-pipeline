import os
import json
import random

def create_jsonl(input_folder, output_file, n):
    # Get all JSON files in the directory
    all_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
    
    if len(all_files) == 0:
        print("No JSON files found in the directory.")
        return
    
    # If n is greater than available files, adjust
    n = min(n, len(all_files))
    
    # Randomly sample n files (or use all_files[:n] if you want first n instead)
    selected_files = random.sample(all_files, n)
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for file_name in selected_files:
            file_path = os.path.join(input_folder, file_name)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                image_path = data.get("image_path")
                prediction = data.get("prediction", {})
                
                captions = {
                    "short_caption": prediction.get("short_caption"),
                    "medium_caption": prediction.get("medium_caption"),
                    "long_caption": prediction.get("long_caption"),
                    "visual_caption": prediction.get("visual_caption"),
                    "semantic_caption": prediction.get("semantic_caption")
                }
                
                output_entry = {
                    "image_path": image_path,
                    "caption": captions
                }
                
                # Write as JSONL (one JSON per line)
                out_f.write(json.dumps(output_entry, ensure_ascii=False) + "\n")
            
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
    
    print(f"JSONL file created at: {output_file}")


if __name__ == "__main__":

    input_folder = "/fsxvision_new/pratyush.jena/Datasets/Indic-Laion/vlm_outputs_10"
    output_file = "test_dataset.jsonl"

    #Number of images you want
    n = 200

    create_jsonl(input_folder, output_file, n)


