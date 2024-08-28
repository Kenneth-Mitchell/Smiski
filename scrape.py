import argparse
import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin

def download_images(url):
    # Set up headers to mimic a browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    # Send request to the URL
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    # Create a directory to save images
    os.makedirs("smiski_images", exist_ok=True)

    # Find image tags
    for img in soup.find_all("img"):
        img_url = img.get("src")
        if img_url and "icon" not in img_url and "bottom" not in img_url and "top" not in img_url:
            # Convert relative URLs to absolute URLs
            img_url = urljoin(url, img_url)
            try:
                # Download image
                img_response = requests.get(img_url)
                img_name = img_url.split("/")[-1]
                img_path = os.path.join("smiski_images", img_name)
                
                # Save image
                with open(img_path, "wb") as f:
                    f.write(img_response.content)
                
                print(f"Downloaded {img_name}")
            except Exception as e:
                print(f"Could not download {img_url}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download images from a given URL.")
    parser.add_argument("url", help="The URL to scrape images from.")
    args = parser.parse_args()

    download_images(args.url)

if __name__ == "__main__":
    main()
