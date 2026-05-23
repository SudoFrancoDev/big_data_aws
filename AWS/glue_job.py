import sys
import re
import ast
import unicodedata
import logging
import boto3
import pandas as pd
from io import StringIO

from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext

from langdetect import detect, LangDetectException as Lang
from deep_translator import GoogleTranslator

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────
# Argumentos del job
# ─────────────────────────────────────────────
args = getResolvedOptions(sys.argv, [
    "JOB_NAME",
    "S3_INPUT_BUCKET",
    "S3_INPUT_KEY",
    "S3_OUTPUT_BUCKET",
    "S3_OUTPUT_PREFIX",
    "DB_HOST",
    "DB_PORT",
    "DB_NAME",
    "DB_USER",
    "DB_PASSWORD",
])

sc          = SparkContext()
glueContext = GlueContext(sc)
job         = Job(glueContext)
job.init(args["JOB_NAME"], args)

# ─────────────────────────────────────────────
# Funciones SERIALIZABLES para RDD
# ─────────────────────────────────────────────

def parse_tag_string(tag_str) -> str:
    if isinstance(tag_str, list):
        return " ".join([str(t) for t in tag_str])
    if isinstance(tag_str, str):
        try:
            parsed = ast.literal_eval(tag_str)
            return " ".join([str(t) for t in parsed])
        except (ValueError, SyntaxError):
            return ""
    return ""


def remove_emojis(text: str) -> str:
    return re.sub(r'[^\x00-\x7F]+', ' ', text)


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("utf-8")


def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = normalize_unicode(text)
    text = remove_emojis(text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_row(row: dict):
    taglist_clean = parse_tag_string(row.get("article_tags", ""))
    title         = clean_text(str(row.get("title", "")))
    feed_tag      = clean_text(str(row.get("feed_tag", "")))
    taglist_clean = clean_text(taglist_clean)

    if not title or not feed_tag:
        return None

    return {
        "link":          row.get("link", ""),
        "title":         title,
        "feed_tag":      feed_tag,
        "Taglist_clean": taglist_clean,
        "source":        row.get("source", "medium"),
        "published":     row.get("published", ""),
        "ingestion_date":row.get("ingestion_date", ""),
    }


def normalize_dt(published_str) -> str:
    """
    Normaliza fecha a formato ISO consistente.
    Garantiza que el mismo valor siempre produzca
    el mismo string para comparación y lookup.
    """
    try:
        dt = pd.to_datetime(published_str, utc=True)
        # Formato fijo: "YYYY-MM-DD HH:MM:SS+00:00"
        return dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
    except Exception:
        return ""


def parse_date(published_str) -> dict:
    try:
        dt = pd.to_datetime(published_str, utc=True)
        return {
            "published_dt": dt.strftime("%Y-%m-%d %H:%M:%S+00:00"),
            "year":         int(dt.year),
            "month":        int(dt.month),
            "day":          int(dt.day),
            "day_of_week":  dt.day_name()
        }
    except Exception:
        return None


# ─────────────────────────────────────────────
# Funciones NO serializables (Pandas)
# ─────────────────────────────────────────────

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Lang:
        return "unknown"


def translate_if_needed(text: str, target_lang: str = "en"):
    if not text or not text.strip():
        return None
    try:
        detected = detect_language(text)
        if detected == "unknown":
            return None
        if detected == target_lang:
            return text
        translator = GoogleTranslator(source="auto", target=target_lang)
        translated = translator.translate(text)
        if not translated or not translated.strip():
            return None
        return translated
    except Exception:
        return None


def translate_dataframe(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    for col in columns:
        logger.info(f"Traduciendo columna: {col}")
        df[col] = df[col].apply(lambda x: translate_if_needed(x))
    before = len(df)
    df = df.dropna(subset=columns).reset_index(drop=True)
    logger.info(f"Filas eliminadas por traducción fallida: {before - len(df)}")
    return df


# ─────────────────────────────────────────────
# Utilidades RDS
# ─────────────────────────────────────────────

def load_existing_links(engine) -> set:
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT article_id FROM fact_articles"))
            existing = {row[0] for row in result}
        logger.info(f"Links existentes en RDS: {len(existing)}")
        return existing
    except Exception as e:
        logger.warning(f"No se pudo cargar links: {e}")
        return set()


def load_existing_values(engine, table: str, column: str) -> set:
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT {column} FROM {table}"))
            existing = {row[0] for row in result}
        logger.info(f"Valores existentes en {table}.{column}: {len(existing)}")
        return existing
    except Exception as e:
        logger.warning(f"No se pudo cargar {table}.{column}: {e}")
        return set()


def load_existing_dates_normalized(engine) -> set:
    """
    Carga fechas existentes en dim_date normalizadas
    al formato fijo "YYYY-MM-DD HH:MM:SS+00:00".
    """
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT published_dt FROM dim_date"))
            existing = {normalize_dt(str(row[0])) for row in result}
        logger.info(f"Fechas existentes normalizadas en dim_date: {len(existing)}")
        return existing
    except Exception as e:
        logger.warning(f"No se pudo cargar dim_date: {e}")
        return set()


def get_max_key(engine, table: str, key_column: str) -> int:
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text(f"SELECT COALESCE(MAX({key_column}), 0) FROM {table}")
            )
            max_key = int(result.scalar())
        logger.info(f"Max key en {table}.{key_column}: {max_key}")
        return max_key
    except Exception as e:
        logger.warning(f"No se pudo obtener max key de {table}: {e}")
        return 0


def load_full_lookup(engine, table, value_col, key_col) -> dict:
    """
    Carga lookup completo normalizando las claves
    al mismo formato que usamos en el pipeline.
    """
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT {value_col}, {key_col} FROM {table}"))
            # Normalizar fechas si es la tabla dim_date
            if table == "dim_date":
                return {normalize_dt(str(row[0])): int(row[1]) for row in result}
            return {row[0]: int(row[1]) for row in result}
    except Exception as e:
        logger.warning(f"No se pudo cargar lookup de {table}: {e}")
        return {}


def save_to_s3(s3_client, df: pd.DataFrame, bucket: str, key: str):
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    s3_client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    logger.info(f"Guardado en s3://{bucket}/{key} ({len(df)} registros)")


def save_to_rds(engine, df: pd.DataFrame, table: str, conflict_column: str):
    from sqlalchemy import text
    cols         = ", ".join(df.columns)
    placeholders = ", ".join([f":{col}" for col in df.columns])
    stmt = text(f"""
        INSERT INTO {table} ({cols})
        VALUES ({placeholders})
        ON CONFLICT ({conflict_column}) DO NOTHING
    """)
    with engine.connect() as conn:
        for _, row in df.iterrows():
            conn.execute(stmt, row.to_dict())
        conn.commit()
    logger.info(f"Insertados en RDS tabla '{table}': {len(df)} registros")


# ═════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═════════════════════════════════════════════

# ─────────────────────────────────────────────
# 0. Conectar a RDS
# ─────────────────────────────────────────────
from sqlalchemy import create_engine

engine = create_engine(
    f"postgresql+psycopg2://{args['DB_USER']}:{args['DB_PASSWORD']}"
    f"@{args['DB_HOST']}:{args['DB_PORT']}/{args['DB_NAME']}"
)

# ─────────────────────────────────────────────
# 1. Leer CSV desde S3 → RDD
# ─────────────────────────────────────────────
logger.info(f"Leyendo datos desde s3://{args['S3_INPUT_BUCKET']}/{args['S3_INPUT_KEY']}")

s3_path  = f"s3://{args['S3_INPUT_BUCKET']}/{args['S3_INPUT_KEY']}"
spark_df = glueContext.spark_session.read.option("header", "true").csv(s3_path)
raw_rdd  = spark_df.rdd.map(lambda row: row.asDict())

total_raw = raw_rdd.count()
logger.info(f"Total registros en S3: {total_raw}")

# ─────────────────────────────────────────────
# 2. Filtrar solo registros NUEVOS
# ─────────────────────────────────────────────
logger.info("Filtrando registros nuevos...")

existing_links    = load_existing_links(engine)
existing_links_bc = sc.broadcast(existing_links)

new_rdd   = raw_rdd.filter(lambda r: r.get("link", "") not in existing_links_bc.value)
total_new = new_rdd.count()
logger.info(f"Registros nuevos: {total_new} / {total_raw}")

if total_new == 0:
    logger.info("No hay registros nuevos. El job finaliza sin cambios.")
    job.commit()
    sys.exit(0)

# ─────────────────────────────────────────────
# 3. Limpiar con RDD.map()
# ─────────────────────────────────────────────
logger.info("Limpiando datos con RDD.map()...")

cleaned_rdd   = new_rdd.map(clean_row).filter(lambda x: x is not None)
total_cleaned = cleaned_rdd.count()
logger.info(f"Registros limpios: {total_cleaned}")

# ─────────────────────────────────────────────
# 4. Traducir con Pandas
# ─────────────────────────────────────────────
logger.info("Traduciendo con Pandas...")

df = pd.DataFrame(cleaned_rdd.collect())
df = translate_dataframe(df, ["title", "feed_tag", "Taglist_clean"])

total_transformed = len(df)
logger.info(f"Registros tras traducción: {total_transformed}")

if total_transformed == 0:
    logger.info("No quedaron registros válidos.")
    job.commit()
    sys.exit(0)

# ─────────────────────────────────────────────
# 5. Normalizar fechas en Pandas
# Con formato fijo antes de volver al RDD
# ─────────────────────────────────────────────
df["published_dt_normalized"] = df["published"].apply(normalize_dt)

# Verificar normalización
sample = df[["published", "published_dt_normalized"]].head(3)
logger.info(f"Muestra de fechas normalizadas:\n{sample.to_string()}")

# ─────────────────────────────────────────────
# 6. Volver a RDD
# ─────────────────────────────────────────────
transformed_rdd = sc.parallelize(df.to_dict("records"))
transformed_rdd.cache()

# ─────────────────────────────────────────────
# 7. Max keys y valores existentes
# ─────────────────────────────────────────────
logger.info("Obteniendo max keys y valores existentes...")

max_source_key   = get_max_key(engine, "dim_source",   "source_key")
max_feed_tag_key = get_max_key(engine, "dim_feed_tag", "feed_tag_key")
max_tag_key      = get_max_key(engine, "dim_tag",      "tag_key")
max_date_key     = get_max_key(engine, "dim_date",     "date_key")

existing_sources_bc   = sc.broadcast(load_existing_values(engine, "dim_source",   "source"))
existing_feed_tags_bc = sc.broadcast(load_existing_values(engine, "dim_feed_tag", "feed_tag"))
existing_tags_bc      = sc.broadcast(load_existing_values(engine, "dim_tag",      "tag_name"))
existing_dates_bc     = sc.broadcast(load_existing_dates_normalized(engine))

# ─────────────────────────────────────────────
# 8. Construir dimensiones con RDDs
# ─────────────────────────────────────────────
logger.info("Construyendo dimensiones...")

source_rdd = (
    transformed_rdd
    .map(lambda r: r["source"])
    .distinct()
    .filter(lambda x: x not in existing_sources_bc.value)
    .zipWithIndex()
    .map(lambda x: {"source": x[0], "source_key": int(x[1]) + max_source_key + 1})
)

feed_tag_rdd = (
    transformed_rdd
    .map(lambda r: r["feed_tag"])
    .distinct()
    .filter(lambda x: x not in existing_feed_tags_bc.value)
    .zipWithIndex()
    .map(lambda x: {"feed_tag": x[0], "feed_tag_key": int(x[1]) + max_feed_tag_key + 1})
)

tag_rdd = (
    transformed_rdd
    .map(lambda r: r["Taglist_clean"])
    .distinct()
    .filter(lambda x: x not in existing_tags_bc.value)
    .zipWithIndex()
    .map(lambda x: {"tag_name": x[0], "tag_key": int(x[1]) + max_tag_key + 1})
)

date_rdd = (
    transformed_rdd
    .map(lambda r: r.get("published_dt_normalized", ""))
    .distinct()
    .filter(lambda x: x and x not in existing_dates_bc.value)
    .map(parse_date)
    .filter(lambda x: x is not None)
    .zipWithIndex()
    .map(lambda x: {**x[0], "date_key": int(x[1]) + max_date_key + 1})
)

# ─────────────────────────────────────────────
# 9. Construir fact_articles con lookups
# Cargar lookups completos desde RDS con
# claves normalizadas al mismo formato
# ─────────────────────────────────────────────
logger.info("Construyendo fact_articles...")

full_source_lookup   = load_full_lookup(engine, "dim_source",   "source",       "source_key")
full_feed_tag_lookup = load_full_lookup(engine, "dim_feed_tag", "feed_tag",     "feed_tag_key")
full_tag_lookup      = load_full_lookup(engine, "dim_tag",      "tag_name",     "tag_key")
full_date_lookup     = load_full_lookup(engine, "dim_date",     "published_dt", "date_key")  # ← ya normalizado

# Agregar nuevos registros a los lookups
for r in source_rdd.collect():
    full_source_lookup[r["source"]] = r["source_key"]
for r in feed_tag_rdd.collect():
    full_feed_tag_lookup[r["feed_tag"]] = r["feed_tag_key"]
for r in tag_rdd.collect():
    full_tag_lookup[r["tag_name"]] = r["tag_key"]
for r in date_rdd.collect():
    # Normalizar la clave al mismo formato fijo
    full_date_lookup[normalize_dt(r["published_dt"])] = r["date_key"]

# Log para verificar lookup de fechas
logger.info(f"Total entradas en date_lookup: {len(full_date_lookup)}")
sample_dates = list(full_date_lookup.items())[:3]
logger.info(f"Muestra date_lookup: {sample_dates}")

source_lookup_bc   = sc.broadcast(full_source_lookup)
feed_tag_lookup_bc = sc.broadcast(full_feed_tag_lookup)
tag_lookup_bc      = sc.broadcast(full_tag_lookup)
date_lookup_bc     = sc.broadcast(full_date_lookup)

fact_rdd = transformed_rdd.map(lambda r: {
    "article_id":   r.get("link", ""),
    "title":        r.get("title", ""),
    "source_key":   source_lookup_bc.value.get(r.get("source")),
    "feed_tag_key": feed_tag_lookup_bc.value.get(r.get("feed_tag")),
    "tag_key":      tag_lookup_bc.value.get(r.get("Taglist_clean")),
    "date_key":     date_lookup_bc.value.get(r.get("published_dt_normalized")),  # ← formato fijo
})

# Debug — verificar nulls
null_dates = fact_rdd.filter(lambda r: r["date_key"] is None).count()
logger.info(f"Registros con date_key null: {null_dates}")

# ─────────────────────────────────────────────
# 10. Convertir RDDs a DataFrames
# ─────────────────────────────────────────────
logger.info("Convirtiendo RDDs a DataFrames...")

dim_source   = pd.DataFrame(source_rdd.collect())
dim_feed_tag = pd.DataFrame(feed_tag_rdd.collect())
dim_tag      = pd.DataFrame(tag_rdd.collect())
dim_date     = pd.DataFrame(date_rdd.collect())
fact         = pd.DataFrame(fact_rdd.collect())

logger.info(f"  dim_source:    {len(dim_source)} nuevos")
logger.info(f"  dim_feed_tag:  {len(dim_feed_tag)} nuevos")
logger.info(f"  dim_tag:       {len(dim_tag)} nuevos")
logger.info(f"  dim_date:      {len(dim_date)} nuevos")
logger.info(f"  fact_articles: {len(fact)} nuevos")

# ─────────────────────────────────────────────
# 11. Guardar en S3
# ─────────────────────────────────────────────
logger.info("Guardando en S3...")

s3     = boto3.client("s3")
PREFIX = args["S3_OUTPUT_PREFIX"]
BUCKET = args["S3_OUTPUT_BUCKET"]

for name, table_df in {
    "dim_source":    dim_source,
    "dim_feed_tag":  dim_feed_tag,
    "dim_tag":       dim_tag,
    "dim_date":      dim_date,
    "fact_articles": fact,
}.items():
    if not table_df.empty:
        save_to_s3(s3, table_df, BUCKET, f"{PREFIX}/{name}_incremental.csv")

# ─────────────────────────────────────────────
# 12. Guardar en RDS PostgreSQL
# ─────────────────────────────────────────────
logger.info("Guardando en RDS PostgreSQL...")

try:
    if not dim_source.empty:
        save_to_rds(engine, dim_source,   "dim_source",    "source_key")
    if not dim_feed_tag.empty:
        save_to_rds(engine, dim_feed_tag, "dim_feed_tag",  "feed_tag_key")
    if not dim_tag.empty:
        save_to_rds(engine, dim_tag,      "dim_tag",       "tag_key")
    if not dim_date.empty:
        save_to_rds(engine, dim_date,     "dim_date",      "date_key")
    if not fact.empty:
        save_to_rds(engine, fact,         "fact_articles", "article_id")

    logger.info("Guardado incremental en RDS exitoso")

except Exception as e:
    logger.error(f"Error guardando en RDS: {e}")
    raise

# ─────────────────────────────────────────────
# Finalizar job
# ─────────────────────────────────────────────
transformed_rdd.unpersist()
for bc in [existing_links_bc, existing_sources_bc, existing_feed_tags_bc,
           existing_tags_bc, existing_dates_bc, source_lookup_bc,
           feed_tag_lookup_bc, tag_lookup_bc, date_lookup_bc]:
    bc.unpersist()

job.commit()
logger.info(f"Glue Job completado. {total_transformed} nuevos registros procesados.")