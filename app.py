import io
import json
import os

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB max
app.secret_key = os.urandom(24)

ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls", "json", "tsv", "parquet"}

# In-memory store (keyed by session label)
dataframes = {}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def safe_val(v):
    """Convert numpy types to native Python for JSON serialisation."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if pd.isna(v) if not isinstance(v, (list, dict)) else False:
        return None
    return v


def analyze_dataframe(df: pd.DataFrame) -> dict:
    analysis = {
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "duplicates": int(df.duplicated().sum()),
        "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
        "columns": [],
    }

    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        col_info = {
            "name": col,
            "dtype": str(df[col].dtype),
            "null_count": null_count,
            "null_pct": round(null_count / max(len(df), 1) * 100, 2),
            "unique_count": int(df[col].nunique()),
            "sample_values": [safe_val(v) for v in df[col].dropna().head(3).tolist()],
            "stats": None,
        }

        if pd.api.types.is_numeric_dtype(df[col]):
            col_info["stats"] = {
                "min": safe_val(df[col].min()),
                "max": safe_val(df[col].max()),
                "mean": safe_val(round(df[col].mean(), 4)) if not df[col].empty else None,
                "std": safe_val(round(df[col].std(), 4)) if not df[col].empty else None,
                "median": safe_val(df[col].median()),
            }

        analysis["columns"].append(col_info)

    return analysis


def df_to_preview(df: pd.DataFrame, n: int = 10) -> list:
    preview_df = df.head(n).copy()
    # Convert datetimes to strings for JSON
    for col in preview_df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        preview_df[col] = preview_df[col].astype(str)
    return [{k: safe_val(v) for k, v in row.items()} for row in preview_df.to_dict("records")]


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    filename = secure_filename(file.filename)
    ext = filename.rsplit(".", 1)[1].lower()

    try:
        read_map = {
            "csv": lambda f: pd.read_csv(f),
            "tsv": lambda f: pd.read_csv(f, sep="\t"),
            "xlsx": lambda f: pd.read_excel(f),
            "xls": lambda f: pd.read_excel(f),
            "json": lambda f: pd.read_json(f),
            "parquet": lambda f: pd.read_parquet(f),
        }
        df = read_map[ext](file)

        dataframes["original"] = df.copy()
        dataframes["current"] = df.copy()
        dataframes.pop("processed", None)

        return jsonify({
            "success": True,
            "filename": filename,
            "analysis": analyze_dataframe(df),
            "preview": df_to_preview(df),
            "columns": list(df.columns),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/auto-clean", methods=["POST"])
def auto_clean():
    if "current" not in dataframes:
        return jsonify({"error": "No data loaded"}), 400

    df = dataframes["current"].copy()
    report = []

    # 1. Strip whitespace from string columns
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        df[col] = df[col].str.strip()
    if len(obj_cols):
        report.append(f"Stripped whitespace from {len(obj_cols)} text column(s).")

    # 2. Drop entirely-empty columns
    empty_cols = [c for c in df.columns if df[c].isnull().all()]
    if empty_cols:
        df.drop(columns=empty_cols, inplace=True)
        report.append(f"Removed {len(empty_cols)} fully-empty column(s): {', '.join(empty_cols)}.")

    # 3. Drop duplicate rows
    before = len(df)
    df.drop_duplicates(inplace=True)
    removed = before - len(df)
    if removed:
        report.append(f"Removed {removed} duplicate row(s).")

    # 4. Fill numeric nulls with median
    for col in df.select_dtypes(include=["number"]).columns:
        n = int(df[col].isnull().sum())
        if n:
            med = df[col].median()
            df[col] = df[col].fillna(med)
            report.append(f"Filled {n} null(s) in '{col}' with median {med:.4g}.")

    # 5. Fill categorical nulls with mode / 'Unknown'
    for col in df.select_dtypes(include=["object"]).columns:
        n = int(df[col].isnull().sum())
        if n:
            modes = df[col].mode()
            fill = modes.iloc[0] if len(modes) else "Unknown"
            df[col] = df[col].fillna(fill)
            report.append(f"Filled {n} null(s) in '{col}' with '{fill}'.")

    # 6. Infer better numeric/datetime types
    for col in df.select_dtypes(include=["object"]).columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() / max(len(df), 1) > 0.9:
            df[col] = converted
            report.append(f"Inferred numeric type for '{col}'.")
            continue
        try:
            dt = pd.to_datetime(df[col], errors="coerce")
            if dt.notna().sum() / max(len(df), 1) > 0.9:
                df[col] = dt
                report.append(f"Inferred datetime type for '{col}'.")
        except Exception:
            pass

    if not report:
        report.append("Dataset already looks clean — no changes needed.")

    dataframes["current"] = df
    dataframes["processed"] = df.copy()

    return jsonify({
        "success": True,
        "report": report,
        "analysis": analyze_dataframe(df),
        "preview": df_to_preview(df),
        "columns": list(df.columns),
    })


@app.route("/process", methods=["POST"])
def process():
    if "current" not in dataframes:
        return jsonify({"error": "No data loaded"}), 400

    df = dataframes["current"].copy()
    ops = request.json.get("operations", [])
    report = []

    for op in ops:
        t = op.get("type")

        if t == "drop_nulls":
            before = len(df)
            cols = op.get("columns") or None
            df = df.dropna(subset=cols) if cols else df.dropna()
            report.append(f"Dropped rows with nulls — removed {before - len(df)} row(s).")

        elif t == "fill_nulls":
            strategy = op.get("strategy", "mean")
            cols = op.get("columns") or df.columns.tolist()
            for col in cols:
                if col not in df.columns:
                    continue
                s = df[col]
                if strategy == "mean" and pd.api.types.is_numeric_dtype(s):
                    df[col] = s.fillna(s.mean())
                elif strategy == "median" and pd.api.types.is_numeric_dtype(s):
                    df[col] = s.fillna(s.median())
                elif strategy == "mode":
                    m = s.mode()
                    df[col] = s.fillna(m.iloc[0] if len(m) else "Unknown")
                elif strategy == "zero":
                    df[col] = s.fillna(0)
                elif strategy == "unknown":
                    df[col] = s.fillna("Unknown")
                elif strategy == "forward":
                    df[col] = s.ffill()
                elif strategy == "backward":
                    df[col] = s.bfill()
            report.append(f"Filled nulls using '{strategy}' strategy on {len(cols)} column(s).")

        elif t == "drop_duplicates":
            before = len(df)
            df = df.drop_duplicates()
            report.append(f"Removed {before - len(df)} duplicate row(s).")

        elif t == "drop_columns":
            cols = [c for c in op.get("columns", []) if c in df.columns]
            df.drop(columns=cols, inplace=True)
            report.append(f"Dropped column(s): {', '.join(cols)}.")

        elif t == "rename_columns":
            renames = {k: v for k, v in op.get("renames", {}).items() if k in df.columns}
            df.rename(columns=renames, inplace=True)
            report.append(f"Renamed {len(renames)} column(s).")

        elif t == "strip_whitespace":
            for col in df.select_dtypes(include=["object"]).columns:
                df[col] = df[col].str.strip()
            report.append("Stripped whitespace from all text columns.")

        elif t == "lowercase":
            cols = op.get("columns") or df.select_dtypes(include=["object"]).columns.tolist()
            for col in cols:
                if col in df.columns:
                    df[col] = df[col].str.lower()
            report.append(f"Lowercased {len(cols)} text column(s).")

        elif t == "convert_type":
            col = op.get("column")
            dtype = op.get("dtype")
            if col and dtype and col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                    report.append(f"Converted '{col}' → {dtype}.")
                except Exception as exc:
                    report.append(f"Failed to convert '{col}' → {dtype}: {exc}.")

    dataframes["current"] = df
    dataframes["processed"] = df.copy()

    return jsonify({
        "success": True,
        "report": report,
        "analysis": analyze_dataframe(df),
        "preview": df_to_preview(df),
        "columns": list(df.columns),
    })


@app.route("/reset", methods=["POST"])
def reset():
    if "original" not in dataframes:
        return jsonify({"error": "No original data found"}), 400
    df = dataframes["original"].copy()
    dataframes["current"] = df
    dataframes.pop("processed", None)
    return jsonify({
        "success": True,
        "analysis": analyze_dataframe(df),
        "preview": df_to_preview(df),
        "columns": list(df.columns),
    })


@app.route("/download")
def download():
    fmt = request.args.get("format", "csv")
    df = dataframes.get("processed") or dataframes.get("current")
    if df is None:
        return jsonify({"error": "No data to download"}), 400

    buf = io.BytesIO()
    if fmt == "csv":
        df.to_csv(buf, index=False)
        buf.seek(0)
        return send_file(buf, mimetype="text/csv", as_attachment=True, download_name="cleaned_data.csv")
    elif fmt == "xlsx":
        df.to_excel(buf, index=False)
        buf.seek(0)
        return send_file(
            buf,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            as_attachment=True,
            download_name="cleaned_data.xlsx",
        )
    elif fmt == "json":
        df.to_json(buf, orient="records", indent=2)
        buf.seek(0)
        return send_file(buf, mimetype="application/json", as_attachment=True, download_name="cleaned_data.json")
    else:
        return jsonify({"error": "Unsupported format"}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5001)



