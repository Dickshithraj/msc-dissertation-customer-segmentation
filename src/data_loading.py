"""
Data loading module for the Online Retail II dataset.

The source workbook contains two sheets covering different calendar years.
Both sheets share an identical schema, so they are loaded separately, tagged
with a ``Year`` column to preserve provenance, and concatenated into a single
DataFrame that all downstream pipeline stages consume.

Keeping loading strictly separate from cleaning means each concern can be
tested, replaced, or re-run independently.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config import RAW_EXCEL_PATH, SHEET_2009_2010, SHEET_2010_2011

logger = logging.getLogger(__name__)

# Mapping of sheet name → value written into the ``Year`` column.
_SHEET_YEAR_MAP: dict[str, str] = {
    SHEET_2009_2010: "2009-2010",
    SHEET_2010_2011: "2010-2011",
}

# Explicit dtypes avoid silent coercion and speed up Excel parsing.
_DTYPE_MAP: dict[str, str] = {
    "Invoice": "str",
    "StockCode": "str",
    "Description": "str",
    "Country": "str",
}


def _load_sheet(sheet_name: str, year_label: str) -> pd.DataFrame:
    """Read one sheet from the raw Excel workbook and tag it with ``year_label``.

    Parameters
    ----------
    sheet_name:
        Exact name of the Excel sheet to read, as stored in ``config.py``.
    year_label:
        Human-readable string written into the ``Year`` column so downstream
        code can filter by source period if needed.

    Returns
    -------
    pd.DataFrame
        Raw, un-cleaned rows from the requested sheet with a ``Year`` column
        appended.
    """
    logger.info("Loading sheet '%s' from %s", sheet_name, RAW_EXCEL_PATH)
    df = pd.read_excel(
        RAW_EXCEL_PATH,
        sheet_name=sheet_name,
        dtype=_DTYPE_MAP,
    )
    df["Year"] = year_label
    logger.info("  -> %d rows loaded.", len(df))
    return df


def load_raw_data() -> pd.DataFrame:
    """Load and concatenate both annual sheets of the Online Retail II workbook.

    The two sheets cover 01/12/2009–09/12/2010 and 01/12/2010–09/12/2011
    respectively. Combining them gives a full two-year transaction history,
    which is the minimum horizon recommended for stable RFM and CLV estimates.

    Returns
    -------
    pd.DataFrame
        Combined raw DataFrame with columns:
        Invoice, StockCode, Description, Quantity, InvoiceDate, Price,
        Customer ID, Country, Year.
        The index is reset so it is contiguous across both sheets.
    """
    frames = [
        _load_sheet(sheet_name, year_label)
        for sheet_name, year_label in _SHEET_YEAR_MAP.items()
    ]
    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "Combined dataset: %d rows across %d sheets.", len(combined), len(frames)
    )
    return combined
