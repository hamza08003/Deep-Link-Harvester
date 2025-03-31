import os
import logging
import imagehash
from PIL import Image
from rich.table import Table
from rich.theme import Theme
from rich.console import Console
from typing import Tuple, List, Dict, Optional


# configure logging for debugging details
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")

# create a console with a custom theme if desired.
custom_theme = Theme({
    "retained": "bold green",
    "deleted": "bold red",
    "error": "bold yellow"
})
console = Console(theme=custom_theme)


# it is a dictionary that stores hash values as keys and lists of image file paths as values.
# Each key in hash_map is a tuple containing:
#   - Perceptual hash (pHash)
#   - Average hash (aHash)
# Each value is a list of image file paths that are considered duplicates based on similarity.
# Example: { (pHash, aHash): [imagePath1, imagePath2, ...], (pHash2, aHash2): [imagePath3, ...] }

hash_map: Dict[Tuple[str, str], List[str]] = {}


def get_image_hash(image_path: str, hash_size: int = 8) -> Tuple[str, str]:
    """
    Compute perceptual and average hashes for an image.

    Args:
        image_path (str): Path to the image file.
        hash_size (int, optional): Size of the hash (default is 8). A larger hash gives more accuracy.

    Returns:
        Tuple[str, str]: A tuple containing the perceptual hash (pHash) and average hash (aHash).
    """
    with Image.open(image_path) as img:
        phash = imagehash.phash(img, hash_size=hash_size)
        ahash = imagehash.average_hash(img, hash_size=hash_size)
    return (str(phash), str(ahash))  # return both hashes as a tuple


def get_image_size(image_path: str) -> Tuple[float, int, int]:
    """
    Get the file size in KB and image dimensions (width, height).

    Args:
        image_path (str): Path to the image file.

    Returns:
        Tuple[float, int, int]: File size in KB, width, and height of the image.
    """
    file_size_kb = os.path.getsize(image_path) / 1024  # convert bytes to KB
    with Image.open(image_path) as img:
        width, height = img.size 
    return file_size_kb, width, height


def hamming_distance(hash1: str, hash2: str) -> int:
    """
    Compute the Hamming distance between two hash strings.

    Args:
        hash1 (str): First hash string.
        hash2 (str): Second hash string.

    Returns:
        int: Number of differing bits between the two hash strings.
    """
    distance = 0
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            distance += 1
    return distance


def process_images(image_dir: str, similarity_threshold: int = 5) -> None:
    """
    Process images in a directory, group similar ones, and retain only the highest quality version.

    Steps:
    1. Compute hashes for all images and group similar ones.
    2. Within each group, retain only the largest image (highest resolution).
    3. Delete smaller/duplicate images.
    4. Display a detailed summary table for each group including image size and dimensions using Rich.

    Args:
        image_dir (str): Path to the directory containing images.
        similarity_threshold (int, optional): Maximum allowed Hamming distance for images to be considered duplicates. Default is 5.

    Returns:
        None: This function directly modifies the filesystem (deletes duplicate images).
    """
    
    logging.info("Starting Image(s) Processing...")
    
    # get list of all image files in the directory
    image_paths = []
    for f in os.listdir(image_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(image_dir, f))
    
    # Step 1: Hash images and group similar ones.
    logging.info("Hashing images and grouping similar ones...")
    
    # loop through each image in the 'images_paths' list and compute its hash
    for image_path in image_paths:
        try:
            # compute hashes for the image
            phash, ahash = get_image_hash(image_path)
            
            # try to find an existing similar hash in the hash_map
            found_match = None
            for stored_hash in hash_map.keys():
                p_dist = hamming_distance(phash, stored_hash[0])
                a_dist = hamming_distance(ahash, stored_hash[1])
                
                # check if the image is similar to an existing one
                if p_dist <= similarity_threshold and a_dist <= similarity_threshold:
                    found_match = stored_hash
                    break
            
            if found_match:
                # add the image to the existing group if found_match in hash_map
                hash_map[found_match].append(image_path)
                logging.debug(f"Grouped {image_path} with group {found_match}")
            else:
                # if no match found, create a new group with the current image
                hash_map[(phash, ahash)] = [image_path]
                logging.debug(f"Created new group for {image_path}")
                
        except Exception as e:
            logging.error(f"Error processing {image_path}: {e}")

    # Structure to hold detailed results per group.
    # Each group will have a list of tuples: (image_path, action, size, dimensions)
    detailed_summary: List[List[Tuple[str, str, float, str]]] = []

    # Step 2: Select the best image (largest file size) in each group and remove the rest.
    logging.info("Selecting best images and removing duplicates...")

    # loop through each group in the hash_map to determine the best image to retain
    for group_hash, paths in hash_map.items():
        group_summary = []
        # if only one image in the group, retain it and record its details.
        if len(paths) == 1:
            size, width, height = get_image_size(paths[0])
            dimensions = f"{width}x{height}"
            group_summary.append((paths[0], "Retained", size, dimensions))
        # if multiple images in the group, determine the best one by file size.
        else:
            best_image = max(paths, key=lambda img: get_image_size(img)[0])
            size_best, width_best, height_best = get_image_size(best_image)
            dimensions_best = f"{width_best}x{height_best}"
            group_summary.append((best_image, "Retained", size_best, dimensions_best))

            # delete duplicates after recording each duplicate's info.
            for img in paths:
                if img != best_image:
                    # get duplicate image details before deletion.
                    size_dup, width_dup, height_dup = get_image_size(img)
                    dimensions_dup = f"{width_dup}x{height_dup}"
                    try:
                        os.remove(img)
                        group_summary.append((img, "Deleted", size_dup, dimensions_dup))
                        logging.debug(f"Deleted duplicate {img}")
                    except Exception as e:
                        logging.error(f"Error deleting {img}: {e}")
                        group_summary.append((img, "Error", size_dup, dimensions_dup))
        detailed_summary.append(group_summary)


    # Step 3: Print the detailed summary table for each group using Rich.
    console.rule("[bold blue]Detailed Summary Table[/bold blue]")
    for idx, group in enumerate(detailed_summary, start=1):
        table = Table(title=f"Group {idx}", header_style="bold magenta")
        table.add_column("No.", style="dim", width=5)
        table.add_column("Image", width=40)
        table.add_column("Action", width=10, justify="center")
        table.add_column("Size (KB)", justify="right", width=10)
        table.add_column("Dimensions", justify="center", width=15)
        
        for i, (img, action, size, dimensions) in enumerate(group, start=1):
            if action == "Retained":
                action_text = "[retained]✓ Retained[/retained]"
            elif action == "Deleted":
                action_text = "[deleted]✗ Deleted[/deleted]"
            else:
                action_text = "[error]! Error[/error]"
            table.add_row(str(i), os.path.basename(img), action_text, f"{size:.2f}", dimensions)
        console.print(table)
    console.rule("[bold blue]Processing complete![/bold blue]")
    console.print("Retained images are marked with [green]✓[/green] and deleted ones with [red]✗[/red].")


# Example Usage
images_dir = "Downloaded_Images//Depth Level 0"
process_images(images_dir)
