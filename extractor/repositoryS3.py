import pandas as pd
import boto3
import logging
from typing import List, Dict
from io import StringIO


logger = logging.getLogger(__name__)


class MediumS3Repository:
    def __init__(self, bucket_name: str, key: str):
        self.bucket_name = bucket_name
        self.key = key
        self.s3 = boto3.client("s3")

    def save_incremental(self, rows: List[Dict]) -> pd.DataFrame:
        """
        Guarda artículos incrementalmente en S3 evitando duplicados.
        """

        if not rows:
            raise ValueError("No hay datos para guardar")

        df_new = pd.DataFrame(rows)

        if "link" not in df_new.columns:
            raise KeyError("Column 'link' is required")

        logger.info(f"Saving {len(df_new)} new records to S3")

        try:
            # 🔹 Intentar leer archivo existente en S3
            try:
                obj = self.s3.get_object(
                    Bucket=self.bucket_name,
                    Key=self.key
                )

                df_old = pd.read_csv(obj["Body"])

                df_final = (
                    pd.concat([df_old, df_new])
                    .drop_duplicates(subset=["link"])
                )

            except self.s3.exceptions.NoSuchKey:
                logger.info("No existing file found in S3. Creating new one.")
                df_final = df_new

            # 🔹 Guardar en memoria (buffer)
            csv_buffer = StringIO()
            df_final.to_csv(csv_buffer, index=False)

            # 🔹 Subir a S3
            self.s3.put_object(
                Bucket=self.bucket_name,
                Key=self.key,
                Body=csv_buffer.getvalue()
            )

            logger.info(f"Upload successful. Total records: {len(df_final)}")

            return df_final

        except Exception as e:
            raise RuntimeError(
                f"Error guardando datos en S3: {self.bucket_name}/{self.key}"
            ) from e