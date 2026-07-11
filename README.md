# AeroDocs-RAG

Retrieve instructions from aircraft maintenance manual for a given query

## ollama requirements
- make sure ollama has the mentioned model and is running
- ollama pull model_name
- ollama serve
- after usage
- ollama rm model_name

## start unvicorn 
`uvicorn api.main:app --reload`

## send a curl request
```
curl -X POST \                                            
http://localhost:8000/api/v1/query \
-H "Content-Type: application/json" \
-d '{
"question": "How do I inspect the bleed air valve?"
}'
```
