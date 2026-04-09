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
# TagProcessor
# ─────────────────────────────────────────────
class TagProcessor:
    def __init__(self, dataframe: pd.DataFrame, tag_column: str = "article_tags"):
        self.df = dataframe
        self.tag_column = tag_column

    def _parse_tag_string(self, tag_str) -> list:
        if isinstance(tag_str, list):
            return [str(tag) for tag in tag_str]
        if isinstance(tag_str, str):
            try:
                return [str(tag) for tag in ast.literal_eval(tag_str)]
            except (ValueError, SyntaxError):
                return []
        return []

    def clean_tags(self) -> pd.DataFrame:
        self.df[self.tag_column] = self.df[self.tag_column].fillna("")
        parsed_tags = self.df[self.tag_column].apply(self._parse_tag_string)
        self.df["Taglist_clean"] = parsed_tags.apply(lambda tags: " ".join(tags))
        return self.df


# ─────────────────────────────────────────────
# TextCleaner
# ─────────────────────────────────────────────
class TextCleaner:
    def __init__(self, dataframe: pd.DataFrame, text_columns: list):
        missing = [col for col in text_columns if col not in dataframe.columns]
        if missing:
            raise ValueError(f"Columns not found in dataframe: {missing}")
        self.translator = GoogleTranslator(source="auto", target="en")
        self.df = dataframe
        self.text_columns = text_columns

    def _remove_emojis(self, text: str) -> str:
        return re.sub(r'[^\x00-\x7F]+', ' ', text)

    def _normalize_unicode(self, text: str) -> str:
        text = unicodedata.normalize("NFKD", text)
        return text.encode("ascii", "ignore").decode("utf-8")

    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = self._normalize_unicode(text)
        text = self._remove_emojis(text)
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def clean(self) -> pd.DataFrame:
        for col in self.text_columns:
            self.df[col] = self.df[col].fillna("").astype(str).apply(self._clean_text)
        return self.df

    def _detect_language(self, text: str) -> str:
        try:
            return detect(text)
        except Lang:
            return "unknown"

    def _translate_if_needed(self, text: str, target_lang: str) -> str:
        if not text.strip():
            return text
        if self._detect_language(text) != target_lang:
            try:
                return self.translator.translate(text)
            except Exception as e:
                logger.warning(f"Error traduciendo texto: {e}")
                return text
        return text

    def normalize_language(self, target_lang: str = "en") -> pd.DataFrame:
        for col in self.text_columns:
            self.df[col] = self.df[col].apply(
                lambda x: self._translate_if_needed(x, target_lang)
            )
        return self.df


# ─────────────────────────────────────────────
# Utilidad: guardar DataFrame en S3
# ─────────────────────────────────────────────
def save_to_s3(s3_client, df: pd.DataFrame, bucket: str, key: str):
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    s3_client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    logger.info(f"Guardado en s3://{bucket}/{key} ({len(df)} registros)")


# ─────────────────────────────────────────────
# Utilidad: insertar DataFrame en RDS
# ─────────────────────────────────────────────
def save_to_rds(engine, df: pd.DataFrame, table: str, conflict_column: str):
    from sqlalchemy import text
    cols = ", ".join(df.columns)
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


# ─────────────────────────────────────────────
# 1. Leer CSV desde S3
# ─────────────────────────────────────────────
logger.info(f"Leyendo datos desde s3://{args['S3_INPUT_BUCKET']}/{args['S3_INPUT_KEY']}")

s3  = boto3.client("s3")
obj = s3.get_object(Bucket=args["S3_INPUT_BUCKET"], Key=args["S3_INPUT_KEY"])
df  = pd.read_csv(obj["Body"])

logger.info(f"Registros leídos: {len(df)}")

# ─────────────────────────────────────────────
# 2. Transformaciones de texto
# ─────────────────────────────────────────────

# Paso 1: Limpiar y parsear tags
logger.info("Procesando tags...")
df = TagProcessor(df, tag_column="article_tags").clean_tags()

# Paso 2: Limpiar texto
logger.info("Aplicando limpieza de texto...")
cleaner = TextCleaner(df, ["title", "feed_tag", "Taglist_clean"])
df = cleaner.clean()

# Paso 3: Traducir al inglés
logger.info("Normalizando idioma a inglés...")
df = cleaner.normalize_language(target_lang="en")

logger.info(f"Transformación de texto completada. Registros: {len(df)}")

# ─────────────────────────────────────────────
# 3. Modelo dimensional (Data Warehouse)
# ─────────────────────────────────────────────
logger.info("Construyendo modelo dimensional...")

# ── dim_source ────────────────────────────────
dim_source = df[["source"]].drop_duplicates().reset_index(drop=True)
dim_source["source_key"] = dim_source.index + 1

# ── dim_feed_tag ──────────────────────────────
dim_feed_tag = df[["feed_tag"]].drop_duplicates().reset_index(drop=True)
dim_feed_tag["feed_tag_key"] = dim_feed_tag.index + 1

# ── dim_tag ───────────────────────────────────
dim_tag = df[["Taglist_clean"]].drop_duplicates().reset_index(drop=True)
dim_tag = dim_tag.rename(columns={"Taglist_clean": "tag_name"})
dim_tag["tag_key"] = dim_tag.index + 1

# ── dim_date ──────────────────────────────────
df["published_dt"] = pd.to_datetime(df["published"], utc=True, errors="coerce")
dim_date = df[["published_dt"]].drop_duplicates().reset_index(drop=True)
dim_date["date_key"]    = dim_date.index + 1
dim_date["year"]        = dim_date["published_dt"].dt.year
dim_date["month"]       = dim_date["published_dt"].dt.month
dim_date["day"]         = dim_date["published_dt"].dt.day
dim_date["day_of_week"] = dim_date["published_dt"].dt.day_name()

# ── fact_articles ─────────────────────────────
fact = df.copy()
fact = fact.merge(dim_source,   on="source",                              how="left")
fact = fact.merge(dim_feed_tag, on="feed_tag",                            how="left")
fact = fact.merge(dim_tag,      left_on="Taglist_clean", right_on="tag_name", how="left")
fact = fact.merge(dim_date,     on="published_dt",                        how="left")
fact = fact.rename(columns={"link": "article_id"})
fact = fact[["article_id", "title", "source_key", "feed_tag_key", "tag_key", "date_key"]]

logger.info("Modelo dimensional construido exitosamente")
logger.info(f"  dim_source:   {len(dim_source)} registros")
logger.info(f"  dim_feed_tag: {len(dim_feed_tag)} registros")
logger.info(f"  dim_tag:      {len(dim_tag)} registros")
logger.info(f"  dim_date:     {len(dim_date)} registros")
logger.info(f"  fact_articles:{len(fact)} registros")

# ─────────────────────────────────────────────
# 4. Guardar en S3
# ─────────────────────────────────────────────
PREFIX = args["S3_OUTPUT_PREFIX"]  # ej: "medium/transformed"
BUCKET = args["S3_OUTPUT_BUCKET"]

tables = {
    "dim_source":    dim_source,
    "dim_feed_tag":  dim_feed_tag,
    "dim_tag":       dim_tag,
    "dim_date":      dim_date,
    "fact_articles": fact,
}

for name, table_df in tables.items():
    save_to_s3(s3, table_df, BUCKET, f"{PREFIX}/{name}.csv")

# ─────────────────────────────────────────────
# 5. Guardar en RDS PostgreSQL
# ─────────────────────────────────────────────
logger.info("Guardando modelo dimensional en RDS...")

try:
    from sqlalchemy import create_engine

    engine = create_engine(
        f"postgresql+psycopg2://{args['DB_USER']}:{args['DB_PASSWORD']}"
        f"@{args['DB_HOST']}:{args['DB_PORT']}/{args['DB_NAME']}"
    )

    save_to_rds(engine, dim_source,   "dim_source",    "source_key")
    save_to_rds(engine, dim_feed_tag, "dim_feed_tag",  "feed_tag_key")
    save_to_rds(engine, dim_tag,      "dim_tag",       "tag_key")
    save_to_rds(engine, dim_date,     "dim_date",      "date_key")
    save_to_rds(engine, fact,         "fact_articles", "article_id")

    logger.info("Guardado en RDS exitoso")

except Exception as e:
    logger.error(f"Error guardando en RDS: {e}")
    raise

# ─────────────────────────────────────────────
# Finalizar job
# ─────────────────────────────────────────────
job.commit()
logger.info("Glue Job completado exitosamente")