# /data_pipeline — Zepto Data & AI Platform

Scrapes book catalogue data from `books.toscrape.com`, cleans it, converts
price to INR using a fixed baseline rate, loads it into a normalized SQLite
database, and queries it with both SQL and pandas.

## Setup

```bash
pip install -r requirements.txt
```

## Run (in order)

```bash
python scrape.py            # -> raw_books.csv
python clean_and_load.py    # -> books.db
python queries.py           # -> prints all query results + verification
```

## Design decisions

- **Category scoping**: categories are discovered dynamically from the site's
  sidebar navigation rather than hard-coded, and the scraper walks full
  pagination within each category until it has covered at least 3 categories
  and 60+ books. This keeps the scraper resilient to catalogue changes.
- **Missing/malformed fields**: if `star_rating` or `price` fails to parse for
  a row, the pipeline does **not** crash — the numeric field is imputed with
  the column median (chosen over dropping rows, since a single bad field
  shouldn't cost an otherwise-valid book's title/category/availability data).
- **Currency conversion**: `price_inr` uses the required fixed, project-defined
  rate **1 GBP = 105.50 INR** — a constant for this assignment, not a live or
  historical market rate. No API call or network lookup is used for this
  required path.
- **Schema**: `categories(category_id PK, category_name)` and
  `books(book_id PK, title, price_gbp, price_inr, rating, in_stock,
  category_id FK -> categories.category_id)` — a standard 1:N normalized
  relationship.
- **Verification**: the JOIN query result is independently reproduced with
  `pd.merge` on in-memory DataFrames (no SQL) and compared row-for-row against
  the SQL JOIN output to demonstrate both approaches agree.
