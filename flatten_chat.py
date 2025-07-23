import json

# Load your nested JSON
with open("_chat.json", "r", encoding="utf-8") as f:
    nested_data = json.load(f)

# Flatten the data
flat_data = []
for date, times in nested_data.items():
    for time, messages in times.items():
        for message in messages:
            flat_data.append({
                "date": date,
                "time": time.replace('\u202f', ' ').strip(),  # Replace narrow no-break space
                "sender": message.get("sender"),
                "message": message.get("message")
            })

# Save to a new file
with open("_chat_flat.json", "w", encoding="utf-8") as f:
    json.dump(flat_data, f, indent=2, ensure_ascii=False)

print("✅ Saved to _chat_flat.json")
