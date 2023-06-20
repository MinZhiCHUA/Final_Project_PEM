ENDPOINT_ID="2845017123195977728"
PROJECT_ID="artefact-taxonomy"
INPUT_DATA_FILE="scripts/query_att_classifer.json"


curl \
-X POST \
-H "Authorization: Bearer $(gcloud auth print-access-token)" \
-H "Content-Type: application/json" \
https://europe-west1-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/europe-west1/endpoints/${ENDPOINT_ID}:predict \
-d "@${INPUT_DATA_FILE}"
