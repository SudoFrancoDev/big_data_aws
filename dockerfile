# Imagen base oficial de AWS para Lambda con Python 3.12
FROM public.ecr.aws/lambda/python:3.12
 
# Copiar e instalar dependencias
COPY requirements.txt .
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"
 
# Copiar todo el código de la carpeta extractor/
COPY extractor/extractor.py       ${LAMBDA_TASK_ROOT}/
COPY extractor/pipeline.py        ${LAMBDA_TASK_ROOT}/
COPY extractor/repositoryS3.py    ${LAMBDA_TASK_ROOT}/
COPY extractor/lambda_handler.py  ${LAMBDA_TASK_ROOT}/
 
# Punto de entrada: archivo.función
CMD ["lambda_handler.lambda_handler"]