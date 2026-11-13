#!/bin/bash
# Status and last log lines of an AI Studio run: studio_job.sh JOB_ID [LINES]
# The API key is read from ~/.config/ai-studio/key (studio access).
K=$(cat ~/.config/ai-studio/key); S=${AI_STUDIO_URL:-http://10.1.250.48}
curl -s "$S/api/jobs/$1" -H "Authorization: Bearer $K" | python3 -c "import json,sys;d=json.load(sys.stdin);print('status',d['status'],'step',d['step'],'/',d['total_steps'],d.get('error') or '')"
curl -s "$S/api/jobs/$1/logs?limit=${2:-25}" -H "Authorization: Bearer $K" | python3 -c "
import json,sys
for l in json.load(sys.stdin): print(l['level'][:4], l['line'][:240].replace('\n',' | '))"
