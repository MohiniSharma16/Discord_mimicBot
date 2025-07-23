import re
import json
from collections import defaultdict

# Regex for your format: DD/MM/YYYY, HH:MM - Name: Message
msg_pattern = re.compile(r'^(\d{2}/\d{2}/\d{4}), (\d{2}:\d{2}) - (.*?): (.*)')

chat_data = defaultdict(lambda: defaultdict(list))

with open('_chat.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    match = msg_pattern.match(line)

    if match:
        date, time, sender, message = match.groups()
        chat_data[date][time].append({
            "sender": sender,
            "message": message
        })

# Save to JSON
with open('_chat.json', 'w', encoding='utf-8') as f:
    json.dump(chat_data, f, ensure_ascii=False, indent=2)

print("✅ WhatsApp chat successfully converted to 'converted_chat_final.json'")