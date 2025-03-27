import asyncio
import json
import tldextract
import pandas as pd
from urllib.parse import urlparse
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig



def get_main_domain(domain_url: str) -> str:
    """
    Extract the main domain from a given URL, handling alsmost all type of known domain structures.
    
    Args:
        domain_url (str): The URL to extract the main domain from.
    
    Returns:
        str: The main domain (e.g., "example.com" from "sub.example.com").
    """
    parsed = urlparse(domain_url)
    # remove authentication and ports if any
    netloc = parsed.netloc.split('@')[-1].split(':')[0]  
    
    ext = tldextract.extract(netloc)
    if ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return netloc  # return raw if no valid domain is found


def group_links_by_section(domain_url: str, links: list[dict]) -> dict:
    """
    Group internal links by the first segment of their path.
    Links that are alone in their category are moved to the 'root' category.
    Duplicate links are automatically removed (case-insensitive).
    
    Args:
        domain_url (str): The main domain URL.
        links (list[dict]): A list of dictionaries, each containing link information.
        
    Returns:
        dict: A dictionary where keys are the first path segment (or 'root') and
              values are lists of unique URLs under that section.
    """
    main_domain = get_main_domain(domain_url)

    # First pass - group all links using case-insensitive comparison
    temp_grouped = {}
    url_case_map = {}  # maps lowercase URLs to their original case version
    
    for link in links:
        href = link.get("href", "")
        parsed = urlparse(href)
        
        # skip external links
        if main_domain not in parsed.netloc.lower():
            continue
            
        href_lower = href.lower()
        
        # store the first occurrence's version
        if href_lower not in url_case_map:
            url_case_map[href_lower] = href
            
        path_segments = parsed.path.strip("/").split("/")
        section = path_segments[0].lower() if path_segments and path_segments[0] != "" else "root"
        if section not in temp_grouped:
            temp_grouped[section] = set()
        temp_grouped[section].add(href_lower)
    
    # Second pass - move single links to root
    grouped = {}
    for section, urls in temp_grouped.items():
        if len(urls) == 1 and section != "root":
            grouped.setdefault("root", set()).update(urls)
        else:
            grouped[section] = urls
    
    # convert sets back to sorted lists, using the original case versions
    return {k: sorted([url_case_map[url] for url in v]) for k, v in grouped.items()}


async def harvest_links(domain_url: str, browser_config: BrowserConfig, run_config: CrawlerRunConfig) -> None:
    """
    Crawl the specified domain to extract internal links, group them by section,
    and save the results into a JSON file.
    
    Parameters:
        domain_url (str): The URL of the domain to crawl.
        browser_config (BrowserConfig): Configuration settings for the crawler browser.
        run_config (CrawlerRunConfig): Run configuration for the crawler.
    """
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=domain_url, config=run_config)

        if not result.success:
            print(f"Failed to crawl: {result.error_message}")
            return

        # extract internal links from the result.
        int_links = result.links.get("internal", [])
        print(f"Total internal links found for {domain_url}: {len(int_links)}")
        # group internal links by the first segment of the URL path.
        grouped_links = group_links_by_section(domain_url, int_links)
        # build the final structure mapping the main domain to its grouped links.
        final_result = {domain_url: grouped_links}

        # call the `save_results` function to save the results to a JSON file
        await save_results(final_result, f"{get_main_domain(domain_url)}_subpages_links.json", "./Crawled Site URLs")


async def load_domains(domains_file: str) -> list[str]:
    """
    Load domains from an xlsx file.
    
    Args:
        domains_file (str): Path to the xlsx file containing domains
        
    Returns:
        list[str]: List of domain URLs
    """
    df = pd.read_excel(domains_file)
    return df['Domain'].str.strip().tolist()


async def save_results(results: dict, filename: str, output_dir: str) -> None:
    """
    Save the results to a JSON file.

    Args:
        results (dict): The results dictionary to save to a JSON file
        filename (str): The name of the file to save the results to
        output_dir (str): The path to the output file
    
    Returns:
        None - saves the results to a JSON file
    """
    with open(f"{output_dir}/{filename}", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


async def main(domains_file_path: str, browser_config: BrowserConfig, run_config: CrawlerRunConfig):
    """
    Main function to process all domains in the xlsx file.
    
    Args:
        domains_file_path (str): Path to the xlsx file containing the domains
        browser_config (BrowserConfig): Browser configuration
        run_config (CrawlerRunConfig): Crawler configuration
    """
    domains = await load_domains(domains_file_path)
    print(f"\nTotal domains to process: {len(domains)}")
    print("-" * 100)
    for domain in domains:
        print(f"\nProcessing: {domain}")
        try:
            await harvest_links(domain, browser_config, run_config)
        except Exception as e:
            print(f"Error processing {domain}: {str(e)}")
        print("-" * 100)


if __name__ == "__main__":
    # path to the xlsx file containing the domains
    domains_file_path = "./Input Data/domains.xlsx"

    # `crawl4ai` browser config which controls the browser behavior
    browser_config = BrowserConfig(
        text_mode=True,
        light_mode=True,
        user_agent_mode="random"
    )

    # `crawl4ai` run config which controls the crawler behavior
    run_config = CrawlerRunConfig()

    # run the main function
    asyncio.run(main(domains_file_path, browser_config, run_config))
