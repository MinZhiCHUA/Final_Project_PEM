# Final_Project_PEM



# Bento ML
save model

build bentoML:
bentoml build -f ./src/serving_attribute_classification/bentofile.yaml ./src/ --version 0.0.1
bentoml build -f ./src/serving_attribute_classification/bentofile.yaml ./src/

check model list 
bentoml models list

try to run the service

bentoml serve simplon_attribute_classifier:latest --reload

# Dockerize

<!-- bentoml containerize simplon_attribute_classifier:latest -->
bentoml containerize simplon_attribute_classifier:0.0.2 -t eu.gcr.io/artefact-taxonomy/simplon_attribute_classifier:0.0.2
## run locally to test
docker run -it --rm -p 3000:3000 eu.gcr.io/artefact-taxonomy/simplon_attribute_classifier:0.0.2 serve --production

# Push to GCP
docker push eu.gcr.io/artefact-taxonomy/simplon_attribute_classifier:0.0.3