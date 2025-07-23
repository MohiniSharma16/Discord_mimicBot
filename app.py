import os
import json
import random
import asyncio
import time
from dotenv import load_dotenv

import discord
from discord.ext import commands
from discord import ui, Interaction

import cohere
import tiktoken

from langchain.docstore.document import Document
from langchain_community.vectorstores.faiss import FAISS
from langchain_cohere import CohereEmbeddings

# ────────────────── CONFIG ──────────────────
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
if not (COHERE_API_KEY and DISCORD_TOKEN):
    raise EnvironmentError("Set COHERE_API_KEY & DISCORD_BOT_TOKEN in .env")

co = cohere.Client(COHERE_API_KEY)
encoding = tiktoken.get_encoding("cl100k_base")

# ───────────── TOKEN & CONTEXT ──────────────
def estimate_tokens(text):
    return len(encoding.encode(text))

def limit_context(lines, max_tokens=1800):
    total, out = 0, []
    for ln in reversed(lines):
        n = estimate_tokens(ln)
        if total + n > max_tokens:
            break
        out.append(ln)
        total += n
    return list(reversed(out))

# ───────────── MESSAGE PROCESSING ───────────
def clean_messages(raw):
    seen, out = set(), []
    for m in raw:
        if not m.get("message") or m["message"] == "<Media omitted>":
            continue
        key = f'{m["sender"]}:{m["message"]}'
        if key in seen:
            continue
        seen.add(key)
        sender = m["sender"].strip()
        out.append({
            "sender": sender,
            "role": sender.split()[0],
            "message": m["message"].strip()
        })
    return out

def build_vector(msgs):
    docs = [
        Document(page_content=m["message"],
                 metadata={"sender": m["sender"], "role": m["role"]})
        for m in msgs
    ]
    embedder = CohereEmbeddings(
        cohere_api_key=COHERE_API_KEY,
        model="embed-english-v3.0",
        user_agent="discord-whatsapp-mimic"
    )
    texts = [d.page_content for d in docs]
    metas = [d.metadata for d in docs]
    return FAISS.from_texts(texts, embedder, metadatas=metas)

def format_history_for_prompt(history, user_name, role_name, limit=6):
    """
    Format last N history entries as a dialogue snippet for the prompt.
    History entries look like: "You: message" or "Role: message"
    """
    # Take last `limit` entries
    hist = history[-limit:]
    lines = []
    for entry in hist:
        if entry.startswith("You:"):
            lines.append(f"{user_name}: {entry[4:].strip()}")
        elif entry.startswith(f"{role_name}:"):
            lines.append(f"{role_name}: {entry[len(role_name)+1:].strip()}")
        else:
            # fallback just raw
            lines.append(entry)
    return lines

def gen_reply(user_msg, store, history, role, retries=3, delay=4):
    # Find top 8 similar messages by embedding user message
    similar = store.similarity_search(user_msg, k=8, fetch_k=10)
    ctx = [f'{d.metadata["sender"]}: {d.page_content}' for d in similar]

    # Add recent formatted conversation history for style context
    recent_conv = format_history_for_prompt(history, user_name="You", role_name=role, limit=6)
    ctx.extend(recent_conv)

    # Limit total tokens to avoid prompt overflow
    ctx = limit_context(ctx, 1800)

    # Few-shot examples to guide style mimicry
    examples = f"""
You: Hey, how's it going? 😊
{role}: I'm good, thanks! You?

You: When is the meeting?
{role}: Around 3pm, don't be late!

You: Can you send me the report?
{role}: Sure thing, I'll send it shortly.
""".strip()

    prompt = f"""
You are "{role}", replying on WhatsApp. Match the style, wording, emojis, punctuation, and length of the chat excerpts below.
Be concise and natural.

Examples:
{examples}

Chat excerpts:
{chr(10).join(ctx)}

Recent conversation:
{chr(10).join(format_history_for_prompt(history, user_name="You", role_name=role, limit=3)) if history else "No recent conversation."}

User: {user_msg}
{role}:
""".strip()

    for attempt in range(retries):
        try:
            response = co.generate(
                model="command-r-plus",
                prompt=prompt,
                max_tokens=120,
                temperature=0.5,
                k=1,
                stop_sequences=[f"\nYou:", f"\n{role}:"]
            )
            return response.generations[0].text.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise e

# ────────────── SESSION STATE ───────────────
class Session:
    def __init__(self):
        self.vector = None
        self.history = []
        self.roles = []
        self.role = None

sessions: dict[int, Session] = {}

# ────────────── DISCORD UI ──────────────────
class RoleSelect(ui.Select):
    def __init__(self, roles):
        super().__init__(
            placeholder="Choose role…",
            options=[discord.SelectOption(label=r) for r in roles]
        )

    async def callback(self, interaction: Interaction):
        uid = interaction.user.id
        sessions[uid].role = self.values[0]
        await interaction.response.send_message(
            f"✅ Role set to **{self.values[0]}**. Start chatting!",
            ephemeral=True
        )

class RoleView(ui.View):
    def __init__(self, roles, timeout=120):
        super().__init__(timeout=timeout)
        self.add_item(RoleSelect(roles))

# ────────────── DISCORD BOT ─────────────────
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True  # Enable in Discord developer portal
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user} ({bot.user.id})")

@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user or not isinstance(message.channel, discord.DMChannel):
        return

    uid = message.author.id
    sessions.setdefault(uid, Session())
    session = sessions[uid]

    # Handle uploaded WhatsApp chat file (JSON)
    if message.attachments:
        for att in message.attachments:
            if att.filename.endswith(".json"):
                await message.channel.send("⏳ Processing chat…")
                raw_data = json.loads(await att.read())
                cleaned = clean_messages(raw_data)
                if not cleaned:
                    await message.channel.send("❌ No usable lines found in chat.")
                    return
                # Build vector store in a thread to avoid blocking event loop
                session.vector = await asyncio.to_thread(build_vector, cleaned)
                # Extract unique roles for selection
                session.roles = sorted({m["role"] for m in cleaned})
                session.role = None
                session.history = []
                await message.channel.send(
                    embed=discord.Embed(
                        title="Chat processed",
                        description="Select a persona:"
                    ),
                    view=RoleView(session.roles)
                )
                return
        await message.channel.send("❌ Please upload a `.json` WhatsApp export file.")
        return

    # Ensure vector and role are set before chatting
    if not (session.vector and session.role):
        await message.channel.send("📥 Upload your WhatsApp chat JSON and pick a role first.")
        return

    # Add user message to history
    session.history.append(f"You: {message.content}")

    async with message.channel.typing():
        try:
            reply = await asyncio.to_thread(
                gen_reply, message.content, session.vector, session.history, session.role
            )
        except Exception as e:
            await message.channel.send(f"⚠️ Error: {e}")
            return

    # Append bot reply to history
    session.history.append(f"{session.role}: {reply}")

    await message.channel.send(reply)

# ────────────── RUN ─────────────────────────
if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
