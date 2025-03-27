import os
import re
import time
import asyncio
import random
import aiohttp
import aiofiles
import nest_asyncio
import pandas as pd
from tqdm import tqdm
from aiolimiter import AsyncLimiter


# used for running asyncio in Jupyter and Colab notebook environments
# no neeed to use this in a regular Python .py script file
nest_asyncio.apply()


BASE_DOWNLOAD_FOLDER = "Downloaded_Images"
USER_AGENTS_FILE = "user-agents.txt"
os.makedirs(BASE_DOWNLOAD_FOLDER, exist_ok=True)


def load_user_agents(filepath: str) -> list:
    """
    Load user agents from a text file and return as a list

    Args:
        filepath (str): Path to the text file containing user agents
    
    Returns:
        list: List of user agent strings
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            agents = [line.strip() for line in f if line.strip()]
        return agents
    except Exception as e:
        print(f"[ERROR] Loading user agents: {e}")
        return []


# call the `load_user_agents` function to load the user agents from the file
user_agents = load_user_agents(USER_AGENTS_FILE)

# The AsyncLimiter class is used to limit the rate of requests made by the script.
# It allows a certain number of requests per a given time period.
# It accepts two arguments:
#   - max_rate: The maximum number of requests allowed per time period
#   - time_period: The time period in seconds
# In this case, we allow 600 requests per 60 seconds (1 minute) which is equivalent to 10 requests per second.
rate_limiter = AsyncLimiter(600, 60) 


def get_random_user_agent() -> str:
    """
    Return a random user agent from the list of user agents.
    
    Args:
        None
    
    Returns:
        str: Random user agent string
    """
    return random.choice(user_agents)


def create_depth_folder(depth_level_name: str) -> str:
    """
    Create a folder for the given depth level and return its path.

    Args:
        depth_name (str): Name of the depth level
    
    Returns:
        str: Path to the created folder
    """
    folder_path = os.path.join(BASE_DOWNLOAD_FOLDER, depth_level_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def extract_depth_number(sheet_name: str) -> int:
    """
    Extract the depth number from a sheet name like 'Depth Level N', where N is an integer (e.g., 0, 1, 2, ...).

    Args:
        sheet_name (str): Name of the sheet
    
    Returns:
        int: Extracted depth number
    """
    match = re.search(r'\d+', sheet_name)
    return int(match.group()) if match else 0


async def download_image(session: aiohttp.ClientSession, url: str, folder: str, filename: str) -> str:
    """
    Asynchronously download an image from the given URL.

    Implements:
      - Rate limiting using AsyncLimiter
      - Uniform Random delay between defined range (in seconds).
      - Exponential backoff for retries respecting 'Retry-After' header
      - Rotating user-agent header

    Saves the image as <filename>.jpg in the given folder path.
    
    Args:
        session (aiohttp.ClientSession): Aiohttp client session
        url (str): URL of the image to download
        folder (str): Folder to save the downloaded image
        filename (str): Filename to save the image as
    
    Returns:
        str: Path to the downloaded image file
    """
    retries = 3
    backoff_factor = 2

    for attempt in range(retries):
        await asyncio.sleep(random.uniform(1, 3))
        async with rate_limiter:
            try:
                headers = {
                    "User-Agent": get_random_user_agent(),
                    "Connection": "keep-alive",
                }
                async with session.get(url, timeout=10, headers=headers) as response:
                    if response.status == 200:
                        file_path = os.path.join(folder, f"{filename}.jpg")
                        content = await response.read()
                        async with aiofiles.open(file_path, "wb") as f:
                            await f.write(content)
                        return file_path
                    elif response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 1))
                        print(f"[RATE LIMIT] 429 for {url}. Retrying after {retry_after} sec (attempt {attempt+1}/{retries}).")
                        await asyncio.sleep(retry_after)
                    else:
                        print(f"[ERROR] Status {response.status} for URL: {url}")
                        break
            except asyncio.TimeoutError:
                print(f"[ERROR] Timeout downloading {url} (attempt {attempt+1}/{retries}).")
            except aiohttp.ClientConnectorError as e:
                print(f"[ERROR] Connection error for {url}: {e}")
                break
            except Exception as e:
                print(f"[ERROR] Exception downloading {url}: {e}")
        await asyncio.sleep(backoff_factor ** attempt)
    return None


async def download_images_from_urls(urls_info: list, folder: str) -> list:
    """
    Download images from a list of URLs asynchronously using aiohttp.

    Args:
        urls_info (list): List of tuples (row_number, url, filename)
        folder (str): Folder to save the downloaded images
    
    Returns:
        list: List of file paths for downloaded images
    """


    # The TCPConnector class is used to configure the TCP connection settings for the client session.
    # The limit parameter specifies the maximum number of simultaneous connections to make.
    # In Simple terms, it limits the number of concurrent connections to the server.
    connector = aiohttp.TCPConnector(limit=25)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        results = []

        for _, url, filename in urls_info:
            task = asyncio.create_task(download_image(session, url, folder, filename))
            tasks.append(task)

        for task in tqdm(
            asyncio.as_completed(tasks), 
            total=len(tasks), 
            desc="Downloading Images", 
            leave=True, 
            dynamic_ncols=True,
            colour="green"
        ):
            result = await task
            if result is not None:
                results.append(result)

        return results
    

async def process_excel_file(excel_path: str) -> None:
    """
    Reads an Excel file with multiple sheets.
    For each sheet starting with 'Depth Level', extracts the 'Image Address' column,
    generates filenames based on the row number and depth level, and downloads images into a folder.

    Args:
        excel_path (str): Path to the Excel file

    Returns:
        None
    """
    print(f"[INFO] Reading Excel file: {excel_path}")
    sheets = pd.read_excel(excel_path, sheet_name=None)
    total_images_attempted = 0

    for sheet_name, df in sheets.items():
        if not sheet_name.lower().startswith("depth level"):
            continue

        depth_num = extract_depth_number(sheet_name)
        print(f"\n[INFO] Processing sheet: {sheet_name} (Depth {depth_num})")

        if "Image Address" not in df.columns:
            print(f"[WARN] Sheet '{sheet_name}' missing 'Image Address' column. Skipping.....")
            continue

        urls_info = []
        for index, row in df.iterrows():
            if pd.notna(row["Image Address"]):
                row_num = index + 2  # excel row number
                url = row["Image Address"]
                filename = f"{row_num}_d{depth_num}"
                urls_info.append((row_num, url, filename))

        total_images_attempted += len(urls_info)
        print(f"[INFO] Found {len(urls_info)} image URLs in sheet '{sheet_name}'.")
        
        folder = create_depth_folder(sheet_name)
        downloaded_files = await download_images_from_urls(urls_info, folder)
        print(f"[INFO] Downloaded {len(downloaded_files)} images to folder '{folder}'.")

    print(f"\n[INFO] Finished processing. Total images attempted: {total_images_attempted}")


async def main():
    excel_path = input("Enter the path to your Excel file: ").strip()
    if not os.path.exists(excel_path):
        print("[ERROR] Excel file not found!")
        return
    await process_excel_file(excel_path)
    print("\n[INFO] All downloads complete.")


if __name__ == "__main__":
    asyncio.run(main())
