import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import time
import json
import asyncio
import nest_asyncio
import pandas as pd
import tensorflow as tf
from deepface import DeepFace


def check_availability_and_setup_gpu():
    print("\n[SYSTEM] Initializing hardware .....")
    
    # get the list of all available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        print("[SYSTEM] No GPU Available, Using CPU for processing .....")
        return None
    
    print(f"[SYSTEM] Found {len(gpus)} GPU(s) available:")
    for i, gpu in enumerate(gpus):
        print(f"  [{i}] {gpu.name}")
    
    # automatically select if only one GPU available
    if len(gpus) == 1:
        selected_gpu = gpus[0]
        print(f"[SYSTEM] Since There is One GPU Available, Using {selected_gpu.name} By Default:")
    else:
        # let user choose which GPU to use
        while True:
            try:
                choice = input("[SYSTEM] Multiple GPUs found. Enter the index of the GPU to use: ")
                choice_idx = int(choice)
                if 0 <= choice_idx < len(gpus):
                    selected_gpu = gpus[choice_idx]
                    print(f"[SYSTEM] Selected GPU: {selected_gpu.name}")
                    break
                else:
                    print(f"[ERROR] Please enter a number between 0 and {len(gpus)-1}")
            except ValueError:
                print("[ERROR] Please enter a valid number or 'all'")
    

    # configure GPU settings
    try:
        tf.config.experimental.set_memory_growth(selected_gpu, True)
        print(f"[SYSTEM] Enabled memory growth for {selected_gpu.name}")
        
        # tf.config.experimental.set_virtual_device_configuration(
        #     selected_gpu,
        #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        # )
        # print(f"[SYSTEM] Set memory limit for {selected_gpu.name} to 4096MB")

        tf.config.set_visible_devices(selected_gpu, 'GPU')
        print(f"[SYSTEM] Set visible GPU to: {selected_gpu.name}")
        
        return selected_gpu
    
    except RuntimeError as e:
        print(f"[ERROR] GPU configuration error: {e}")
        return None


async def process_single_image(image_path: str, semaphore: asyncio.Semaphore):
    async with semaphore:
        try:
            start_time = time.time()
            print(f"[PROCESS] Starting: {os.path.basename(image_path)}")
            
            results = await asyncio.to_thread(
                DeepFace.represent,
                img_path=image_path,
                model_name="ArcFace",
                detector_backend="retinaface",
                enforce_detection=False,
                align=True
            )
            
            if not results:
                print(f"[WARNING] No faces detected in {os.path.basename(image_path)}")
                return []
            
            elapsed = time.time() - start_time
            print(f"[SUCCESS] Processed {os.path.basename(image_path)} " f"({len(results)} faces, {elapsed:.2f}s)")
            
            return [{
                "image_path": image_path,
                "face_index": idx,
                "embedding": face["embedding"]
            } for idx, face in enumerate(results)]
            
        except Exception as e:
            print(f"[FAILURE] Error processing {os.path.basename(image_path)} " f"({elapsed:.2f}s): {str(e)}")
            return []

async def process_image_batch(image_paths: list, batch_size: int = 3):
    semaphore = asyncio.Semaphore(batch_size)
    tasks = [process_single_image(img, semaphore) for img in image_paths]
    results = []
    for future in asyncio.as_completed(tasks):
        results.extend(await future)
    return results

def save_results(data: list, output_path: str):
    if not data:
        print("[WARNING] No embeddings extracted - empty results")
        return False
    
    try:
        df = pd.DataFrame(data)
        df["embedding"] = df["embedding"].apply(json.dumps)
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"[SUCCESS] Saved {len(df)} face embeddings to {output_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save results: {e}")
        return False

async def main_processing_pipeline(input_directory_filepath: str, output_csv_filepath: str):
    print("\n" + "=" * 70)
    print("[INFO] Starting face detection and embedding extraction pipeline.....")
    print("=" * 70)
    start_time = time.time()
    
    # apply nest_asyncio to allow nested event loops
    nest_asyncio.apply()

    # check GPU availability and setup
    gpu_device = check_availability_and_setup_gpu()
    
    # cllect valid image files from the given input directory path
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = [
        os.path.join(input_directory_filepath, f) for f in os.listdir(input_directory_filepath)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]
    
    if not image_files:
        print("[ERROR] No valid images found in directory")
        return False
    
    print(f"\n[INFO] Found {len(image_files)} images to process")

    print("\n" + "=" * 30)
    print("[INFO] Processing images:")
    print("=" * 30)
    
    # process images in batches
    try:
        embeddings = await process_image_batch(image_files)
        if not embeddings:
            print("[WARNING] No faces detected in any images")
            return False
        
        # save results
        if not save_results(embeddings, output_csv_filepath):
            return False
        
        # final report
        elapsed = time.time() - start_time
        print(f"[SYSTEM] Pipeline completed in {elapsed:.2f} seconds")
        print(f"[SYSTEM] Processed {len(embeddings)} faces from {len(image_files)} images")
        return True
        
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Face Detector and Vector Embedding Extractor")
    parser.add_argument("input_dir_filepath", help="Directory containing face images to process")
    parser.add_argument("output_csv_filepath", help="Output CSV file path")
    parser.add_argument("--batch_size", type=int, default=5, help="Concurrent processing limit")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir_filepath):
        print(f"[ERROR] Input directory does not exist: {args.input_dir_filepath}")
        exit(1)
    
    # run the pipeline
    success = asyncio.run(main_processing_pipeline(
        input_directory_filepath=args.input_dir_filepath,
        output_csv_filepath=args.output_csv_filepath
    ))
    
    exit(0 if success else 1)
