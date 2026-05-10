#!/bin/bash
# Usage: ./benchmark.sh "Qwen3.6-35B-A3B-spec2" "http://127.0.0.1:8081"
#        ./benchmark.sh --list [endpoint]

export LC_NUMERIC=C

if [[ "$1" == "--list" ]]; then
    ENDPOINT="${2:-http://127.0.0.1:8081}"
    echo "========================================================="
    echo " Available Models at: $ENDPOINT"
    echo "========================================================="
    curl -s "${ENDPOINT}/v1/models" | jq -r '.data[].id' || echo "Error: Could not fetch models or jq is missing."
    echo "========================================================="
    exit 0
fi

MODEL="${1:-Qwen3.6-27B}"
ENDPOINT="${2:-http://127.0.0.1:8081}"

echo "========================================================="
echo " Benchmarking Model: $MODEL"
echo " Endpoint: $ENDPOINT"
echo "========================================================="

echo -n "Triggering model reload... "
curl -s "${ENDPOINT}/v1/models?reload=1" > /dev/null && echo "Done." || echo "Failed."

call_api() {
  local turn_name="$1"
  local payload="$2"

  echo -e "\n---> Running $turn_name..."

  local start_ts=$(date +%s.%N)
  local first_token_ts=""
  local temp_file=$(mktemp)

  curl -s -N -X POST "${ENDPOINT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$payload" | while read -r line; do
      
      if [[ -z "$first_token_ts" && "$line" == data:* && "$line" != *"data: [DONE]"* ]]; then
          first_token_ts=$(date +%s.%N)
          echo "$first_token_ts" > "${temp_file}_first"
      fi
      
      if [[ "$line" == data:* && "$line" != *"data: [DONE]"* ]]; then
         content=$(echo "$line" | sed 's/^data: //')
         has_usage=$(echo "$content" | jq -r 'has("usage")' 2>/dev/null)
         if [[ "$has_usage" == "true" ]]; then
            echo "$content" > "${temp_file}_usage"
         fi
      fi
  done
  
  local end_ts=$(date +%s.%N)
  first_token_ts=$(cat "${temp_file}_first" 2>/dev/null)
  if [[ -z "$first_token_ts" ]]; then first_token_ts=$end_ts; fi

  local usage_json=$(cat "${temp_file}_usage" 2>/dev/null)
  local prompt_tokens=$(echo "$usage_json" | jq -r '.usage.prompt_tokens // 0' 2>/dev/null)
  local completion_tokens=$(echo "$usage_json" | jq -r '.usage.completion_tokens // 0' 2>/dev/null)
  
  prompt_tokens=${prompt_tokens:-0}
  completion_tokens=${completion_tokens:-0}

  local ttft=$(echo "$first_token_ts - $start_ts" | bc -l 2>/dev/null)
  local gen_time=$(echo "$end_ts - $first_token_ts" | bc -l 2>/dev/null)

  local p_s=0
  local t_s=0
  if (( $(echo "$ttft > 0" | bc -l) )); then p_s=$(echo "scale=2; $prompt_tokens / $ttft" | bc -l); fi
  if (( $(echo "$gen_time > 0" | bc -l) )); then t_s=$(echo "scale=2; $completion_tokens / $gen_time" | bc -l); fi

  echo "---------------------------------------------------------"
  echo "Prompt Tokens      : $prompt_tokens"
  echo "Completion Tokens  : $completion_tokens"
  printf "Prompt Eval (p/s)  : %.2f tokens/sec (TTFT: %.2fs)\n" "$p_s" "$ttft"
  printf "Generation  (t/s)  : %.2f tokens/sec (Decode: %.2fs)\n" "$t_s" "$gen_time"
  echo "---------------------------------------------------------"
  
  rm -f "$temp_file" "${temp_file}_first" "${temp_file}_usage"
}

# ==============================================================================
# SAFE STRING DEFINITIONS (No bash escaping needed)
# ==============================================================================

SYS_MSG=$(cat <<'END_MSG'
{"role": "system", "content": "You are an automated ORM and API generation agent. Output raw code, JSON, or Markdown exactly as requested without conversational filler."}
END_MSG
)

USER_1=$(cat <<'END_MSG'
{"role": "user", "content": "SCHEMA:\n1. users: id, name, created_at\n2. orders: id, user_id, amount, created_at\n3. products: id, title, price, created_at\n4. shipments: id, order_id, status, created_at\n5. invoices: id, order_id, total, created_at\n6. payments: id, invoice_id, method, created_at\n7. inventory: id, product_id, stock, created_at\n8. reviews: id, product_id, rating, created_at\n\nTask 1: Generate SQLAlchemy declarative base ORM classes for the first 4 tables (users, orders, products, shipments). Use exact identical boilerplate structure for all."}
END_MSG
)

AST_1=$(cat <<'END_MSG'
{"role": "assistant", "content": "class User(Base):\n    __tablename__ = 'users'\n    id = Column(Integer, primary_key=True)\n    name = Column(String)\n    created_at = Column(DateTime)\n\nclass Order(Base):\n    __tablename__ = 'orders'\n    id = Column(Integer, primary_key=True)\n    user_id = Column(Integer)\n    amount = Column(Float)\n    created_at = Column(DateTime)\n\nclass Product(Base):\n    __tablename__ = 'products'\n    id = Column(Integer, primary_key=True)\n    title = Column(String)\n    price = Column(Float)\n    created_at = Column(DateTime)\n\nclass Shipment(Base):\n    __tablename__ = 'shipments'\n    id = Column(Integer, primary_key=True)\n    order_id = Column(Integer)\n    status = Column(String)\n    created_at = Column(DateTime)"}
END_MSG
)

USER_2=$(cat <<'END_MSG'
{"role": "user", "content": "Task 2: Generate the exact same SQLAlchemy ORM classes for the remaining 4 tables (invoices, payments, inventory, reviews)."}
END_MSG
)

AST_2=$(cat <<'END_MSG'
{"role": "assistant", "content": "class Invoice(Base):\n    __tablename__ = 'invoices'\n    id = Column(Integer, primary_key=True)\n    order_id = Column(Integer)\n    total = Column(Float)\n    created_at = Column(DateTime)\n\nclass Payment(Base):\n    __tablename__ = 'payments'\n    id = Column(Integer, primary_key=True)\n    invoice_id = Column(Integer)\n    method = Column(String)\n    created_at = Column(DateTime)\n\nclass Inventory(Base):\n    __tablename__ = 'inventory'\n    id = Column(Integer, primary_key=True)\n    product_id = Column(Integer)\n    stock = Column(Integer)\n    created_at = Column(DateTime)\n\nclass Review(Base):\n    __tablename__ = 'reviews'\n    id = Column(Integer, primary_key=True)\n    product_id = Column(Integer)\n    rating = Column(Integer)\n    created_at = Column(DateTime)"}
END_MSG
)

USER_3=$(cat <<'END_MSG'
{"role": "user", "content": "Task 3: Now register all 8 tables using a tool call. Output a JSON array containing EXACTLY 8 objects. Format:[{\"tool\": \"register\", \"args\": {\"table\": \"<name>\", \"cache\": true, \"auth\": true}}]"}
END_MSG
)

AST_3=$(cat <<'END_MSG'
{"role": "assistant", "content": "[{\"tool\": \"register\", \"args\": {\"table\": \"users\", \"cache\": true, \"auth\": true}}, {\"tool\": \"register\", \"args\": {\"table\": \"orders\", \"cache\": true, \"auth\": true}}, {\"tool\": \"register\", \"args\": {\"table\": \"products\", \"cache\": true, \"auth\": true}}, {\"tool\": \"register\", \"args\": {\"table\": \"shipments\", \"cache\": true, \"auth\": true}}, {\"tool\": \"register\", \"args\": {\"table\": \"invoices\", \"cache\": true, \"auth\": true}}, {\"tool\": \"register\", \"args\": {\"table\": \"payments\", \"cache\": true, \"auth\": true}}, {\"tool\": \"register\", \"args\": {\"table\": \"inventory\", \"cache\": true, \"auth\": true}}, {\"tool\": \"register\", \"args\": {\"table\": \"reviews\", \"cache\": true, \"auth\": true}}]"}
END_MSG
)

USER_4=$(cat <<'END_MSG'
{"role": "user", "content": "Task 4: Finally, generate a Markdown API specification for all 8 tables. Repeat this exact template for each:\n\n### `<table_name>` API\n- `GET /api/<table_name>`: Fetch records\n- `POST /api/<table_name>`: Create record\n- `DELETE /api/<table_name>/:id`: Delete record"}
END_MSG
)

# ==============================================================================
# PAYLOAD CONSTRUCTION
# ==============================================================================

PAYLOAD_1=$(cat <<EOF
{"model": "${MODEL}", "max_tokens": 2048, "stream": true, "stream_options": {"include_usage": true},
 "messages":[$SYS_MSG, $USER_1]}
EOF
)
call_api "Turn 1 (Cold Start -> Python ORMs)" "$PAYLOAD_1"

PAYLOAD_2=$(cat <<EOF
{"model": "${MODEL}", "max_tokens": 2048, "stream": true, "stream_options": {"include_usage": true},
 "messages": [$SYS_MSG, $USER_1, $AST_1, $USER_2]}
EOF
)
call_api "Turn 2 (KV Cache Hit -> Repetitive Python ORMs)" "$PAYLOAD_2"

PAYLOAD_3=$(cat <<EOF
{"model": "${MODEL}", "max_tokens": 2048, "stream": true, "stream_options": {"include_usage": true},
 "messages":[$SYS_MSG, $USER_1, $AST_1, $USER_2, $AST_2, $USER_3]}
EOF
)
call_api "Turn 3 (KV Cache Hit -> Repetitive JSON Tool Calls)" "$PAYLOAD_3"

PAYLOAD_4=$(cat <<EOF
{"model": "${MODEL}", "max_tokens": 2048, "stream": true, "stream_options": {"include_usage": true},
 "messages":[$SYS_MSG, $USER_1, $AST_1, $USER_2, $AST_2, $USER_3, $AST_3, $USER_4]}
EOF
)
call_api "Turn 4 (KV Cache Hit -> Repetitive Markdown)" "$PAYLOAD_4"

# ==============================================================================
# END METRICS DUMP
# ==============================================================================
echo -e "\n========================================================="
echo " Final Model Metrics (from Prometheus endpoint)"
echo "========================================================="
# Pull metrics, strip out comments and empty lines
curl -s "${ENDPOINT}/metrics?model=${MODEL}" | grep -v "^#" | awk NF
echo "========================================================="
echo " Note: Your journalctl logs hold the exact N-Gram Spec stats."
echo " Run this to see the acceptance rates:"
echo "   journalctl --user -n 50 -u llama-router | grep 'statistics ngram_mod'"
echo "========================================================="
