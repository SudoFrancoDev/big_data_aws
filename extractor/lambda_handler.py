import logging
from extractor import MediumRSSExtractor
from repositoryS3 import MediumS3Repository
from pipeline import MediumPipeline

# Logging (clave en Lambda)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    try:
        logger.info("Starting Medium pipeline...")

        # 🔹 Config (idealmente desde variables de entorno)
        BASE_RSS = "https://medium.com/feed/tag/"
        TAGS = [
            "basic-income",
            "debt",
            "economy",
            "inflation",
            "stock-market",
            "adoption",
            "children",
            "elder-care",
            "fatherhood",
            "motherhood",
            "parenting",
            "pregnancy",
            "seniors",
            "artificial-intelligence",
            "machine-learning",
            "data-science",
            "neural-networks",
            "computer-vision",
            "software-engineering",
            "python",
            "large-language-models",
            "nlp",
            "voice-assistant",
            "data-engineering",
            "data-visualization",
            "database-design",
            "sql",
            "analytics",
            "anxiety",
            "counseling",
            "grief",
            "life-lessons",
            "self-awareness",
            "stress",
            "therapy",
            "trauma",
            "guided-meditation",
            "journaling",
            "meditation",
            "transcendental-meditation",
            "yoga",
            "atheism",
            "epistemology",
            "ethics",
            "existentialism",
            "metaphysics",
            "morality",
            "philosophy-of-mind",
            "stoicism",
            "career-advice",
            "coaching",
            "goal-setting",
            "morning-routines",
            "pomodoro-technique",
            "time-management",
            "work-life-balance",
            "dating",
            "divorce",
            "friendship",
            "love",
            "marriage",
            "polyamory"
            ]

        BUCKET = "my-bucket-proyecto"
        KEY = "medium/processed/medium_articles_data.csv"

        # 🔹 Componentes
        extractor = MediumRSSExtractor(BASE_RSS, TAGS)
        repository = MediumS3Repository(bucket_name=BUCKET, key=KEY)

        pipeline = MediumPipeline(
            extractor=extractor,
            repository=repository
        )

        # 🔹 Ejecutar pipeline
        df_result = pipeline.run()

        logger.info(f"Pipeline finished. Total records: {len(df_result)}")

        return {
            "statusCode": 200,
            "body": f"Procesados {len(df_result)} registros"
        }

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)

        return {
            "statusCode": 500,
            "body": str(e)
        }