"""
STEP 3 - QUERY
Runs 5 required SQL queries against books.db (covering SELECT/WHERE, ORDER BY,
LIMIT, DISTINCT, IN/BETWEEN, and a JOIN), then cross-checks the join query
using pd.read_sql vs. pd.merge on in-memory DataFrames.

Run:
    python queries.py
"""

import sqlite3

import pandas as pd

DB_PATH = "books.db"


def run(cur, label, sql):
    print(f"\n--- {label} ---\n{sql.strip()}\n")
    rows = cur.execute(sql).fetchall()
    cols = [d[0] for d in cur.description]
    out = pd.DataFrame(rows, columns=cols)
    print(out.to_string(index=False))
    return out


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 1) SELECT / WHERE / ORDER BY / LIMIT
    q1 = """
        SELECT title, price_gbp, rating
        FROM books
        WHERE in_stock = 1
        ORDER BY price_gbp DESC
        LIMIT 10;
    """
    run(cur, "Q1: Top 10 most expensive in-stock books", q1)

    # 2) DISTINCT
    q2 = """
        SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name;
    """
    q2_result = run(cur, "Q2: Distinct category names", q2)

    # 3) IN / BETWEEN
    q3 = """
        SELECT title, price_gbp, rating
        FROM books
        WHERE rating IN (4, 5)
          AND price_gbp BETWEEN 10 AND 40
        ORDER BY rating DESC, price_gbp ASC
        LIMIT 15;
    """
    run(cur, "Q3: Highly rated (4-5) books priced 10-40 GBP", q3)

    # 4) JOIN
    q4 = """
        SELECT c.category_name, b.title, b.rating, b.price_inr
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        ORDER BY c.category_name, b.rating DESC
        LIMIT 20;
    """
    q4_result = run(cur, "Q4: Books joined with their category (top rated first)", q4)

    # 5) another SELECT/WHERE/ORDER BY/LIMIT combo
    q5 = """
        SELECT title, category_id, rating
        FROM books
        WHERE in_stock = 0
        ORDER BY rating ASC
        LIMIT 10;
    """
    run(cur, "Q5: Out-of-stock books, lowest rated first", q5)

    # --- pd.read_sql for at least two of the above ---
    print("\n\n=== pd.read_sql verification ===")
    read_sql_q2 = pd.read_sql(q2, conn)
    read_sql_q4 = pd.read_sql(q4, conn)
    print("\npd.read_sql(Q2) matches cursor result:", read_sql_q2.equals(q2_result))
    print("pd.read_sql(Q4) matches cursor result:", read_sql_q4.equals(q4_result))

    # --- pd.merge reproduction of the JOIN query (no SQL) ---
    print("\n\n=== pd.merge reproduction of Q4 (no SQL) ===")
    books_df = pd.read_sql("SELECT * FROM books", conn)
    categories_df = pd.read_sql("SELECT * FROM categories", conn)

    merged = books_df.merge(categories_df, on="category_id", how="inner")
    merged_sorted = (
        merged.sort_values(["category_name", "rating"], ascending=[True, False])
        [["category_name", "title", "rating", "price_inr"]]
        .head(20)
        .reset_index(drop=True)
    )
    sql_sorted = q4_result.reset_index(drop=True)

    print(merged_sorted.to_string(index=False))
    print("\npd.merge result matches SQL JOIN result:", merged_sorted.equals(sql_sorted))

    conn.close()


if __name__ == "__main__":
    main()
