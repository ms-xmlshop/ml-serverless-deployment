curl -m 70 -X POST https://europe-west2-hslibsv2.cloudfunctions.net/digit-2 \
-H "Authorization: bearer $(gcloud auth print-identity-token)" \
-H "Content-Type: application/json" \
-d '{
  "text": "cotton hoodie"
}'

curl -m 70 -X POST https://europe-west2-hslibsv2.cloudfunctions.net/digit-2 -H "Authorization: bearer $(gcloud auth print-identity-token)" -H "Content-Type: application/json" -d '{"text": "cotton hoodie"}'