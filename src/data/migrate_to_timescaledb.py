"""Migrate existing PostgreSQL database to TimescaleDB hypertables.

This script:
1. Checks if TimescaleDB extension is available
2. Installs the extension if possible
3. Converts time-series tables to hypertables (preserving all data)
4. Creates continuous aggregates for expensive queries
5. Sets up compression and retention policies

Safe to run multiple times — all operations use IF NOT EXISTS.

Prerequisites:
- TimescaleDB must be installed in the PostgreSQL server.
  Option A: Use timescale/timescaledb:latest-pg16 Docker image
  Option B: Install timescaledb-2-postgresql-16 package on host

Usage:
    python -m src.data.migrate_to_timescaledb [--dry-run]
"""

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def check_timescaledb(conn) -> bool:
    """Check if TimescaleDB extension is available."""
    cur = conn.cursor()
    try:
        cur.execute("SELECT default_version FROM pg_available_extensions WHERE name = 'timescaledb'")
        row = cur.fetchone()
        if row:
            logger.info(f"TimescaleDB available: version {row[0]}")
            return True
        logger.warning("TimescaleDB extension not found in pg_available_extensions")
        return False
    finally:
        cur.close()


def install_extension(conn) -> bool:
    """Install TimescaleDB extension."""
    cur = conn.cursor()
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")
        logger.info("TimescaleDB extension installed")
        # Verify
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'")
        row = cur.fetchone()
        if row:
            logger.info(f"TimescaleDB version: {row[0]}")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to install TimescaleDB: {e}")
        return False
    finally:
        cur.close()


def is_hypertable(conn, table_name: str) -> bool:
    """Check if a table is already a hypertable."""
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT COUNT(*) FROM timescaledb_information.hypertables "
            "WHERE hypertable_name = %s",
            (table_name,)
        )
        return cur.fetchone()[0] > 0
    except Exception:
        return False
    finally:
        cur.close()


def table_exists(conn, table_name: str) -> bool:
    """Check if a table exists."""
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = %s",
            (table_name,)
        )
        return cur.fetchone()[0] > 0
    finally:
        cur.close()


def drop_pk_constraint(conn, table_name: str, time_col: str):
    """Drop PRIMARY KEY constraint so hypertable conversion can work.

    TimescaleDB requires the time column to be part of any unique constraint.
    Our tables use 'date' as PK — we need to drop it and add a UNIQUE constraint
    that TimescaleDB can work with (it auto-creates the time index).
    """
    cur = conn.cursor()
    try:
        # Find the PK constraint name
        cur.execute("""
            SELECT constraint_name FROM information_schema.table_constraints
            WHERE table_name = %s AND constraint_type = 'PRIMARY KEY'
        """, (table_name,))
        row = cur.fetchone()
        if row:
            pk_name = row[0]
            cur.execute(f"ALTER TABLE {table_name} DROP CONSTRAINT {pk_name}")
            logger.info(f"  Dropped PK constraint '{pk_name}' on {table_name}")
    except Exception as e:
        logger.debug(f"  No PK to drop on {table_name}: {e}")
    finally:
        cur.close()


def convert_timestamp_column(conn, table_name: str):
    """Convert TEXT timestamp column to TIMESTAMPTZ for intraday_bars."""
    cur = conn.cursor()
    try:
        # Check current column type
        cur.execute("""
            SELECT data_type FROM information_schema.columns
            WHERE table_name = %s AND column_name = 'timestamp'
        """, (table_name,))
        row = cur.fetchone()
        if row and row[0] in ('text', 'character varying'):
            logger.info(f"  Converting {table_name}.timestamp from TEXT to TIMESTAMPTZ...")
            cur.execute(f"""
                ALTER TABLE {table_name}
                ALTER COLUMN timestamp TYPE TIMESTAMPTZ
                USING timestamp::TIMESTAMPTZ
            """)
            logger.info(f"  Converted {table_name}.timestamp to TIMESTAMPTZ")
    except Exception as e:
        logger.error(f"  Failed to convert timestamp column on {table_name}: {e}")
    finally:
        cur.close()


# Tables to convert: (table_name, time_column, chunk_interval, extra_unique_cols)
HYPERTABLE_CONFIGS = [
    ("prices",            "date", "1 month",  []),
    ("technicals",        "date", "1 month",  []),
    ("macro",             "date", "1 month",  []),
    ("daily_sentiment",   "date", "1 month",  []),
    ("options_analytics", "date", "1 month",  []),
    ("intraday_features", "date", "1 month",  []),
    ("market_breadth",    "date", "3 months", []),
    ("predictions",       "date", "3 months", []),
    ("performance",       "date", "3 months", []),
    ("backtest_results",  "date", "3 months", []),
    ("intraday_bars",     "timestamp", "1 week", ["ticker"]),
    ("options_chain",     "date", "1 month",  ["contract_symbol"]),
]


def convert_to_hypertable(conn, table_name: str, time_col: str,
                          chunk_interval: str, extra_unique_cols: list,
                          dry_run: bool = False):
    """Convert a regular table to a TimescaleDB hypertable."""
    if not table_exists(conn, table_name):
        logger.info(f"  Table {table_name} does not exist, skipping")
        return False

    if is_hypertable(conn, table_name):
        logger.info(f"  {table_name} is already a hypertable ✓")
        return True

    cur = conn.cursor()
    try:
        # Get row count
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cur.fetchone()[0]
        logger.info(f"  {table_name}: {row_count} rows, converting to hypertable "
                     f"(chunk_interval={chunk_interval})...")

        if dry_run:
            logger.info(f"  [DRY RUN] Would convert {table_name}")
            return True

        # For intraday_bars, convert TEXT timestamp to TIMESTAMPTZ
        if table_name == "intraday_bars":
            convert_timestamp_column(conn, table_name)

        # Convert TEXT date columns to DATE type (required by TimescaleDB)
        if time_col == "date":
            cur.execute("""
                SELECT data_type FROM information_schema.columns
                WHERE table_name = %s AND column_name = 'date'
            """, (table_name,))
            dtype_row = cur.fetchone()
            if dtype_row and dtype_row[0] in ('text', 'character varying'):
                logger.info(f"  Converting {table_name}.date from TEXT to DATE...")
                cur.execute(f"ALTER TABLE {table_name} "
                            f"ALTER COLUMN date TYPE DATE USING date::DATE")

        # Drop PK constraint (TimescaleDB needs time col in unique constraints)
        drop_pk_constraint(conn, table_name, time_col)

        # Add UNIQUE constraint that includes the time column
        unique_cols = [time_col] + extra_unique_cols
        unique_name = f"uq_{table_name}_{'_'.join(unique_cols)}"
        try:
            cols_str = ", ".join(unique_cols)
            cur.execute(f"ALTER TABLE {table_name} ADD CONSTRAINT {unique_name} "
                        f"UNIQUE ({cols_str})")
            logger.info(f"  Added UNIQUE constraint ({cols_str}) on {table_name}")
        except Exception as e:
            logger.debug(f"  Unique constraint may already exist: {e}")
            conn.rollback()
            conn.autocommit = True

        # Convert to hypertable
        cur.execute(
            f"SELECT create_hypertable('{table_name}', '{time_col}', "
            f"chunk_time_interval => INTERVAL '{chunk_interval}', "
            f"if_not_exists => TRUE, migrate_data => TRUE)"
        )
        logger.info(f"  ✓ {table_name} converted to hypertable")
        return True

    except Exception as e:
        logger.error(f"  ✗ Failed to convert {table_name}: {e}")
        try:
            conn.rollback()
            conn.autocommit = True
        except Exception:
            pass
        return False
    finally:
        cur.close()


def create_continuous_aggregates(conn, dry_run: bool = False):
    """Create continuous aggregates for expensive queries."""
    aggregates = [
        # Daily summary from intraday bars
        (
            "cagg_intraday_daily",
            """
            CREATE MATERIALIZED VIEW IF NOT EXISTS cagg_intraday_daily
            WITH (timescaledb.continuous) AS
            SELECT
                time_bucket('1 day', timestamp) AS date,
                ticker,
                first(open, timestamp)  AS open,
                max(high)               AS high,
                min(low)                AS low,
                last(close, timestamp)  AS close,
                sum(volume)             AS volume,
                last(vwap, timestamp)   AS vwap,
                max(high) - min(low)    AS intraday_range,
                count(*)                AS bar_count
            FROM intraday_bars
            GROUP BY time_bucket('1 day', timestamp), ticker
            WITH NO DATA
            """,
            "1 hour", "3 days", "1 hour",
        ),
        # Weekly price aggregates
        (
            "cagg_prices_weekly",
            """
            CREATE MATERIALIZED VIEW IF NOT EXISTS cagg_prices_weekly
            WITH (timescaledb.continuous) AS
            SELECT
                time_bucket('1 week', date) AS week,
                first(open, date)   AS open,
                max(high)            AS high,
                min(low)             AS low,
                last(close, date)    AS close,
                sum(volume)          AS volume,
                avg(close)           AS avg_close
            FROM prices
            GROUP BY time_bucket('1 week', date)
            WITH NO DATA
            """,
            "1 day", "1 month", "1 day",
        ),
        # Monthly price aggregates
        (
            "cagg_prices_monthly",
            """
            CREATE MATERIALIZED VIEW IF NOT EXISTS cagg_prices_monthly
            WITH (timescaledb.continuous) AS
            SELECT
                time_bucket('1 month', date) AS month,
                first(open, date)   AS open,
                max(high)            AS high,
                min(low)             AS low,
                last(close, date)    AS close,
                sum(volume)          AS volume,
                avg(close)           AS avg_close
            FROM prices
            GROUP BY time_bucket('1 month', date)
            WITH NO DATA
            """,
            "1 day", "3 months", "1 day",
        ),
        # Hourly intraday aggregates
        (
            "cagg_intraday_hourly",
            """
            CREATE MATERIALIZED VIEW IF NOT EXISTS cagg_intraday_hourly
            WITH (timescaledb.continuous) AS
            SELECT
                time_bucket('1 hour', timestamp) AS hour,
                ticker,
                first(open, timestamp)  AS open,
                max(high)               AS high,
                min(low)                AS low,
                last(close, timestamp)  AS close,
                sum(volume)             AS volume,
                last(vwap, timestamp)   AS vwap,
                count(*)                AS bar_count
            FROM intraday_bars
            GROUP BY time_bucket('1 hour', timestamp), ticker
            WITH NO DATA
            """,
            "1 hour", "3 days", "1 hour",
        ),
        # Daily options summary
        (
            "cagg_options_daily",
            """
            CREATE MATERIALIZED VIEW IF NOT EXISTS cagg_options_daily
            WITH (timescaledb.continuous) AS
            SELECT
                time_bucket('1 day', date) AS date,
                count(*)                                    AS contract_count,
                sum(volume)                                 AS total_volume,
                sum(open_interest)                          AS total_oi,
                avg(iv)                                     AS avg_iv,
                sum(CASE WHEN option_type = 'put' THEN volume ELSE 0 END)::DOUBLE PRECISION /
                    NULLIF(sum(CASE WHEN option_type = 'call' THEN volume ELSE 0 END), 0)
                                                            AS put_call_volume_ratio,
                sum(CASE WHEN option_type = 'put' THEN open_interest ELSE 0 END)::DOUBLE PRECISION /
                    NULLIF(sum(CASE WHEN option_type = 'call' THEN open_interest ELSE 0 END), 0)
                                                            AS put_call_oi_ratio
            FROM options_chain
            GROUP BY time_bucket('1 day', date)
            WITH NO DATA
            """,
            "1 day", "7 days", "1 day",
        ),
    ]

    for name, create_sql, schedule, start_offset, end_offset in aggregates:
        if dry_run:
            logger.info(f"  [DRY RUN] Would create aggregate: {name}")
            continue
        cur = conn.cursor()
        try:
            # Check if it already exists
            cur.execute(
                "SELECT COUNT(*) FROM timescaledb_information.continuous_aggregates "
                "WHERE view_name = %s", (name,)
            )
            if cur.fetchone()[0] > 0:
                logger.info(f"  {name} already exists ✓")
                cur.close()
                continue

            # Check if source hypertable exists
            source_table = "intraday_bars" if "intraday" in name else \
                           "options_chain" if "options" in name else "prices"
            if not is_hypertable(conn, source_table):
                logger.warning(f"  Skipping {name}: source {source_table} is not a hypertable")
                cur.close()
                continue

            cur.execute(create_sql)
            logger.info(f"  ✓ Created continuous aggregate: {name}")

            # Add refresh policy
            cur.execute(f"""
                SELECT add_continuous_aggregate_policy('{name}',
                    start_offset => INTERVAL '{start_offset}',
                    end_offset   => INTERVAL '{end_offset}',
                    schedule_interval => INTERVAL '{schedule}',
                    if_not_exists => TRUE)
            """)
            logger.info(f"    Refresh policy: every {schedule}, "
                         f"offset {start_offset} → {end_offset}")

            # Initial refresh
            cur.execute(f"CALL refresh_continuous_aggregate('{name}', NULL, NULL)")
            logger.info(f"    Initial refresh complete")

        except Exception as e:
            logger.error(f"  ✗ Failed to create {name}: {e}")
            try:
                conn.rollback()
                conn.autocommit = True
            except Exception:
                pass
        finally:
            cur.close()


def setup_compression(conn, dry_run: bool = False):
    """Set up compression policies for hypertables."""
    compression_configs = [
        # (table, segmentby, orderby, compress_after)
        ("intraday_bars", "ticker", "timestamp DESC", "7 days"),
        ("options_chain", "contract_symbol", "date DESC", "30 days"),
        ("prices", None, "date DESC", "6 months"),
        ("technicals", None, "date DESC", "6 months"),
        ("macro", None, "date DESC", "6 months"),
        ("daily_sentiment", None, "date DESC", "6 months"),
    ]

    for table, segmentby, orderby, compress_after in compression_configs:
        if not is_hypertable(conn, table):
            logger.info(f"  Skipping compression for {table} (not a hypertable)")
            continue

        if dry_run:
            logger.info(f"  [DRY RUN] Would enable compression on {table} "
                         f"(after {compress_after})")
            continue

        cur = conn.cursor()
        try:
            # Check if compression is already enabled
            cur.execute("""
                SELECT compression_enabled
                FROM timescaledb_information.hypertables
                WHERE hypertable_name = %s
            """, (table,))
            row = cur.fetchone()
            if row and row[0]:
                logger.info(f"  {table}: compression already enabled ✓")
                cur.close()
                continue

            # Enable compression
            seg_clause = f", timescaledb.compress_segmentby = '{segmentby}'" if segmentby else ""
            cur.execute(f"""
                ALTER TABLE {table} SET (
                    timescaledb.compress{seg_clause},
                    timescaledb.compress_orderby = '{orderby}'
                )
            """)
            logger.info(f"  ✓ Compression enabled on {table}")

            # Add compression policy
            cur.execute(f"""
                SELECT add_compression_policy('{table}',
                    INTERVAL '{compress_after}',
                    if_not_exists => TRUE)
            """)
            logger.info(f"    Policy: compress after {compress_after}")

        except Exception as e:
            logger.error(f"  ✗ Compression setup failed for {table}: {e}")
            try:
                conn.rollback()
                conn.autocommit = True
            except Exception:
                pass
        finally:
            cur.close()


def setup_retention(conn, dry_run: bool = False):
    """Set up data retention policies."""
    retention_configs = [
        # (table, drop_after) — continuous aggregates preserve summaries
        ("intraday_bars", "1 year"),
    ]

    for table, drop_after in retention_configs:
        if not is_hypertable(conn, table):
            continue

        if dry_run:
            logger.info(f"  [DRY RUN] Would add retention policy on {table}: "
                         f"drop after {drop_after}")
            continue

        cur = conn.cursor()
        try:
            cur.execute(f"""
                SELECT add_retention_policy('{table}',
                    INTERVAL '{drop_after}',
                    if_not_exists => TRUE)
            """)
            logger.info(f"  ✓ Retention policy on {table}: drop after {drop_after}")
        except Exception as e:
            logger.error(f"  ✗ Retention policy failed for {table}: {e}")
        finally:
            cur.close()


def print_status(conn):
    """Print current TimescaleDB status."""
    cur = conn.cursor()
    try:
        # Extension version
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'")
        row = cur.fetchone()
        if row:
            print(f"\nTimescaleDB version: {row[0]}")
        else:
            print("\nTimescaleDB: NOT INSTALLED")
            return

        # Hypertables
        cur.execute("""
            SELECT hypertable_name, num_chunks, compression_enabled
            FROM timescaledb_information.hypertables
            ORDER BY hypertable_name
        """)
        rows = cur.fetchall()
        print(f"\nHypertables ({len(rows)}):")
        for r in rows:
            comp = "compressed" if r[2] else "uncompressed"
            print(f"  {r[0]}: {r[1]} chunks, {comp}")

        # Continuous aggregates
        cur.execute("""
            SELECT view_name FROM timescaledb_information.continuous_aggregates
            ORDER BY view_name
        """)
        rows = cur.fetchall()
        print(f"\nContinuous Aggregates ({len(rows)}):")
        for r in rows:
            print(f"  {r[0]}")

        # Compression stats
        try:
            cur.execute("""
                SELECT hypertable_name,
                       before_compression_total_bytes,
                       after_compression_total_bytes
                FROM timescaledb_information.compression_settings cs
                JOIN hypertable_compression_stats(cs.hypertable_name) hcs ON TRUE
                LIMIT 10
            """)
        except Exception:
            # Simpler query
            cur.execute("""
                SELECT hypertable_name
                FROM timescaledb_information.hypertables
                WHERE compression_enabled = TRUE
            """)
            rows = cur.fetchall()
            if rows:
                print(f"\nCompressed tables: {', '.join(r[0] for r in rows)}")

        # Disk usage
        cur.execute("""
            SELECT hypertable_name,
                   pg_size_pretty(hypertable_size(format('%I.%I',
                       hypertable_schema, hypertable_name)::regclass))
            FROM timescaledb_information.hypertables
            ORDER BY hypertable_name
        """)
        rows = cur.fetchall()
        print(f"\nDisk usage:")
        for r in rows:
            print(f"  {r[0]}: {r[1]}")

    except Exception as e:
        logger.error(f"Status check failed: {e}")
    finally:
        cur.close()


def migrate(dry_run: bool = False):
    """Run the full TimescaleDB migration."""
    from src.data.db_router import DbRouter

    print("=" * 60)
    print("TimescaleDB Migration")
    print("=" * 60)

    router = DbRouter()
    if not router.using_postgres:
        print("ERROR: PostgreSQL not available. TimescaleDB requires PostgreSQL.")
        sys.exit(1)

    conn = router.get_pg()
    conn.autocommit = True

    # Step 1: Check & install extension
    print("\n[1/5] Checking TimescaleDB extension...")
    if not check_timescaledb(conn):
        print("TimescaleDB extension not available in this PostgreSQL installation.")
        print("Options:")
        print("  A) Use Docker image: timescale/timescaledb:latest-pg16")
        print("  B) Install package: apt install timescaledb-2-postgresql-16")
        print("  C) See: https://docs.timescale.com/self-hosted/latest/install/")
        sys.exit(1)

    if not install_extension(conn):
        print("Failed to install TimescaleDB extension.")
        sys.exit(1)

    # Step 2: Convert tables to hypertables
    print("\n[2/5] Converting tables to hypertables...")
    converted = 0
    for table, time_col, chunk_interval, extra_cols in HYPERTABLE_CONFIGS:
        result = convert_to_hypertable(
            conn, table, time_col, chunk_interval, extra_cols, dry_run
        )
        if result:
            converted += 1
    print(f"  {converted}/{len(HYPERTABLE_CONFIGS)} tables converted")

    # Step 3: Create continuous aggregates
    print("\n[3/5] Creating continuous aggregates...")
    create_continuous_aggregates(conn, dry_run)

    # Step 4: Set up compression
    print("\n[4/5] Setting up compression policies...")
    setup_compression(conn, dry_run)

    # Step 5: Set up retention
    print("\n[5/5] Setting up retention policies...")
    setup_retention(conn, dry_run)

    # Print status
    if not dry_run:
        print_status(conn)

    router.close()
    print("\n✓ Migration complete!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Migrate PostgreSQL to TimescaleDB")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without making changes")
    parser.add_argument("--status", action="store_true",
                        help="Show current TimescaleDB status only")
    args = parser.parse_args()

    if args.status:
        from src.data.db_router import DbRouter
        router = DbRouter()
        if router.using_postgres:
            print_status(router.get_pg())
        else:
            print("PostgreSQL not available")
        router.close()
    else:
        migrate(dry_run=args.dry_run)
