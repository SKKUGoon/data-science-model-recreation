from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def create_postgres_engine(dsn: str, **engine_kwargs: Any) -> Engine:
    """
    Create a SQLAlchemy Engine for Postgres from a DSN.

    Examples:
      - postgresql+psycopg2://user:pass@host:5432/dbname
      - postgresql://user:pass@host:5432/dbname
    """
    if not dsn or not isinstance(dsn, str):
        raise ValueError("dsn must be a non-empty string")

    # Sensible defaults; callers can override via engine_kwargs.
    defaults: dict[str, Any] = {
        "pool_pre_ping": True,
    }
    return create_engine(dsn, **{**defaults, **engine_kwargs})


def read_sql_df(
    dsn: str,
    sql: str,
    *,
    params: dict[str, Any] | None = None,
    **engine_kwargs: Any,
) -> pd.DataFrame:
    """
    Read SQL into a DataFrame using pandas.read_sql with a DSN (no hardcoded creds).
    """
    engine = create_postgres_engine(dsn, **engine_kwargs)
    return pd.read_sql(sql, con=engine, params=params)


