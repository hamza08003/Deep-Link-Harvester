import os
import imagehash
from PIL import Image
from typing import Tuple, List, Dict, Optional


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

    Args:
        image_dir (str): Path to the directory containing images.
        similarity_threshold (int, optional): Maximum allowed Hamming distance for images to be considered duplicates. Default is 5.

    Returns:
        None: This function directly modifies the filesystem (deletes duplicate images).
    """
    
    print("\n🔍 Step 1: Hashing Images & Grouping Similar Ones...\n")
    
    # get list of all image files in the directory
    image_paths = []
    for f in os.listdir(image_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(image_dir, f))
    
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
                if p_dist <= similarity_threshold or a_dist <= similarity_threshold:
                    found_match = stored_hash
                    break
            
            if found_match:
                # add the image to the existing group if found_match in hash_map
                hash_map[found_match].append(image_path)
            else:
                # if no match found, create a new group with the current image
                hash_map[(phash, ahash)] = [image_path]
                
        except Exception as e:
            print(f"❌ Error hashing {image_path}: {e}")

    # Step 2: Print hash groups
    print("\n📂 Image Groups Formed:")
    for img_hash, paths in hash_map.items():
        print(f"\n📌 Group: {img_hash}")
        for img in paths:
            size_kb, width, height = get_image_size(img)
            print(f"  ➡ {img} | {size_kb:.2f} KB | {width}x{height}")

    print("\n🛠 Step 2: Keeping Best Images & Removing Duplicates...\n")

    # Step 3: Keep only the highest resolution image in each duplicate group
    for img_hash, paths in hash_map.items():
        if len(paths) > 1:
            # select the largest file (highest resolution assumption)
            largest_image = max(paths, key=lambda img: get_image_size(img)[0])  
            print(f"\n✅ Keeping: {largest_image}")

            # delete smaller images
            for img in paths:
                if img != largest_image:
                    os.remove(img)
                    print(f"❌ Deleted: {img}")

    print("\n🎉 Cleanup Complete! Only high-resolution images remain.")


# example usage
image_dir = "Downloaded_Images//Depth Level 0"
process_images(image_dir)
