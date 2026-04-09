class MediumPipeline:
    def __init__(self, extractor, repository):
        self.extractor = extractor
        self.repository = repository

    def run(self):
        rows = self.extractor.fetch()
        return self.repository.save_incremental(rows)
    