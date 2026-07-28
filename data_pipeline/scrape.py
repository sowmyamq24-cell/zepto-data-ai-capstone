"""
STEP 1 - SCRAPE
Scrapes books.toscrape.com across at least 3 categories and saves the raw
(uncleaned) data to raw_books.csv.

Run:
    python scrape.py
"""

import csv
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "http://books.toscrape.com/"
MIN_BOOKS = 60
MIN_CATEGORIES = 3
OUTPUT_CSV = "raw_books.csv"


def get_soup(url):
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def get_category_links():
    """Return a list of (category_name, category_url) from the homepage sidebar."""
    soup = get_soup(BASE_URL)
    nav = soup.select("div.side_categories ul li ul li a")
    categories = []
    for a in nav:
        name = a.get_text(strip=True)
        url = urljoin(BASE_URL, a["href"])
        categories.append((name, url))
    return categories


def scrape_category(name, url):
    """Scrape every paginated page of a single category. Returns list of dicts."""
    books = []
    next_url = url
    while next_url:
        soup = get_soup(next_url)
        for pod in soup.select("article.product_pod"):
            title = pod.h3.a["title"]
            price_text = pod.select_one("p.price_color").get_text(strip=True)
            rating_text = pod.select_one("p.star-rating")["class"][1]  # e.g. "Three"
            availability_text = pod.select_one("p.instock.availability").get_text(strip=True)
            books.append(
                {
                    "title": title,
                    "price": price_text,
                    "star_rating": rating_text,
                    "availability": availability_text,
                    "category": name,
                }
            )
        next_link = soup.select_one("li.next a")
        next_url = urljoin(next_url, next_link["href"]) if next_link else None
        time.sleep(0.2)  # be polite to the practice site
    return books


def main():
    categories = get_category_links()
    all_books = []
    used_categories = []

    for name, url in categories:
        books = scrape_category(name, url)
        if not books:
            continue
        all_books.extend(books)
        used_categories.append(name)
        print(f"  scraped {len(books):3d} books from category '{name}'")
        # stop once we comfortably clear both thresholds
        if len(all_books) >= MIN_BOOKS and len(used_categories) >= MIN_CATEGORIES:
            break

    print(f"\nTotal books scraped: {len(all_books)} across {len(used_categories)} categories")
    assert len(all_books) >= MIN_BOOKS, "Did not reach the minimum required book count"
    assert len(used_categories) >= MIN_CATEGORIES, "Did not reach the minimum required category count"

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "price", "star_rating", "availability", "category"])
        writer.writeheader()
        writer.writerows(all_books)

    print(f"Saved raw data to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
