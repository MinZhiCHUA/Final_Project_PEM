Model Deployment and API endpoint
Key steps:

Save model to BentoML registry: bin/save_model_attribute_classification.py
Create the Service: src/serving_attribute_classification/service.py
Write the bentofile.yaml
Build the Docker image
VERSION="0.0.1"
GCP_PROJECT="artefact-taxonomy"
MODEL_NAME="attribute_classifier"
SERVICE="${MODEL_NAME}"

YAML_PATH="src/serving_attribute_classification/bentofile.yaml"

IMAGE_URI=eu.gcr.io/$GCP_PROJECT/$SERVICE:$VERSION

bentoml build -f $YAML_PATH ./src/ --version $VERSION

bentoml serve attribute_classifier:latest --production

bentoml containerize $SERVICE:$VERSION -t $IMAGE_URI