try:
    import feedparser
except ImportError as e:
    raise ImportError("Instala feedparser: python -m pip install feedparser") from e

from datetime import datetime, timezone
import logging
import urllib.request

logger = logging.getLogger(__name__)

class MediumRSSExtractor:
    def __init__(self, base_rss: str, tags: list[str]):
        self.base_rss = base_rss
        self.tags = tags
        if not base_rss.startswith("http"):
            raise ValueError("base_rss debe ser una URL válida.")

    def fetch(self) -> list[dict]:
        """
        Extrae artículos desde feeds RSS de Medium.

        Returns
        -------
        list[dict]
            Lista de artículos con metadata normalizada.
        """

        rows = []
        failed = 0

        for feed_tag in self.tags:
            try:
                url = self.base_rss + feed_tag

                # ── Simular navegador para evitar bloqueos de Medium ──
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; RSS reader)"}
                )
                with urllib.request.urlopen(req, timeout=10) as response:
                    raw_content = response.read()

                feed = feedparser.parse(raw_content)

                if feed.bozo:
                    logger.warning(f"Feed con advertencia: {feed_tag} - {feed.bozo_exception}")

                entries = feed.entries
                if not entries:
                    logger.warning(f"Feed vacío (sin entradas): {feed_tag}")
                    failed += 1
                    continue

                for entry in entries:
                    rows.append({
                        "title":          getattr(entry, "title", None),
                        "link":           getattr(entry, "link", None),
                        "published":      getattr(entry, "published", None),
                        "feed_tag":       feed_tag,
                        "article_tags":   [t.term for t in getattr(entry, "tags", [])],
                        "source":         "medium",
                        "ingestion_date": datetime.now(timezone.utc).date().isoformat()
                    })

            except Exception as e:
                logger.error(f"Error al procesar feed: {feed_tag} - {e}")
                failed += 1
                continue

        logger.info(f"Extracción completada. Artículos: {len(rows)} | Feeds fallidos: {failed}")

        if not rows:
            raise RuntimeError(
                f"Ningún feed retornó artículos. "
                f"Total feeds fallidos: {failed}/{len(self.tags)}. "
                f"Medium puede estar bloqueando las peticiones desde AWS."
            )

        return rows