import glob
import os
import tempfile
from collections.abc import Callable

import yt_dlp
from dotenv import load_dotenv
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_STUFF_CHARS = 60_000
MAP_CHUNK_CHARS = 12_000
MAP_CHUNK_OVERLAP = 500
RETRIEVER_CHUNK_CHARS = 1_000
RETRIEVER_CHUNK_OVERLAP = 150
GROQ_WARNING_BYTES = 24 * 1024 * 1024
CHAT_MODEL = "nvidia/nemotron-3.5-lightning:free"
FALLBACK_CHAT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b:free"

SUMMARY_SYSTEM = (
    "You are an expert video summarizer. You produce faithful, specific, and well-organized "
    "summaries from spoken transcripts. Use only information present in the provided transcript "
    "or section summaries. Never invent facts, names, or numbers. Ignore filler words, repeated "
    "phrases, greetings, and sponsor reads. Write in the same language as the transcript."
)

SUMMARY_HUMAN = """Summarize the video transcript below.

<transcript>
{transcript}
</transcript>

Use this structure, adapting the number of bullets to the amount of content:
1. Overview: 2-3 sentences describing what the video is about.
2. Key points: concise bullet points in the order they appear (usually 5-10).
3. Takeaways: actionable bullet points (usually 3-5; skip this section if the video has little content).

Keep the whole summary under 450 words. Do not include timestamps unless they appear in the transcript."""

MAP_SYSTEM = (
    "You are summarizing one section of a longer video transcript. Write 4-8 concise bullet "
    "points that capture the section's key facts, arguments, names, and numbers. Use only "
    "information from the section. Do not add commentary. Write in the same language as the transcript."
)

MAP_HUMAN = """Transcript section:

<section>
{transcript}
</section>"""

REDUCE_SYSTEM = (
    "You are combining section summaries into one final summary of a video. Use only information "
    "from the section summaries. Remove repetition while keeping important details. Never invent "
    "facts, names, or numbers. Write in the same language as the section summaries."
)

REDUCE_HUMAN = """Section summaries:

<summaries>
{summaries}
</summaries>

Use this structure, adapting the number of bullets to the amount of content:
1. Overview: 2-3 sentences describing what the video is about.
2. Key points: concise bullet points in the order they appear (usually 5-10).
3. Takeaways: actionable bullet points (usually 3-5; skip this section if the video has little content).

Keep the whole summary under 450 words."""

QA_SYSTEM = (
    "You answer questions about a video using only the provided transcript excerpts. "
    "If the answer is not in the excerpts, say that you could not find it in the video. "
    "Answer with a complete sentence that includes the key detail, never just a phrase or list of words. "
    "Do not invent details. Use the same language as the question."
)

QA_HUMAN = """Transcript excerpts:

<context>
{context}
</context>

Question: {question}

Answer in one complete sentence with a subject and verb. Do not respond with only a phrase or a list of words."""

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [("system", SUMMARY_SYSTEM), ("human", SUMMARY_HUMAN)]
)
MAP_PROMPT = ChatPromptTemplate.from_messages(
    [("system", MAP_SYSTEM), ("human", MAP_HUMAN)]
)
REDUCE_PROMPT = ChatPromptTemplate.from_messages(
    [("system", REDUCE_SYSTEM), ("human", REDUCE_HUMAN)]
)
QA_PROMPT = ChatPromptTemplate.from_messages(
    [("system", QA_SYSTEM), ("human", QA_HUMAN)]
)


def download_audio(url: str, workdir: str) -> str:
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(workdir, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "playlist_items": "1",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 3,
        "fragment_retries": 3,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if not info:
            raise RuntimeError("yt-dlp did not return any video information.")

        if info.get("_type") == "playlist":
            entries = [entry for entry in (info.get("entries") or []) if entry]
            if not entries:
                raise RuntimeError("No playable video was found at that URL.")
            info = entries[0]

        path = ydl.prepare_filename(info)
        if os.path.isfile(path):
            return path

        candidates = [
            file
            for file in glob.glob(os.path.join(workdir, "*"))
            if os.path.isfile(file) and not file.endswith(".part")
        ]
        if not candidates:
            raise RuntimeError("Audio download failed: no file was produced.")
        return candidates[0]


def transcribe_audio(audio_file: str, client: Groq) -> str:
    size = os.path.getsize(audio_file)
    if size > GROQ_WARNING_BYTES:
        print(
            f"Warning: audio is {size / (1024 * 1024):.1f} MB. "
            "Groq may reject files over about 25 MB."
        )

    with open(audio_file, "rb") as file:
        transcription = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=file,
        )

    text = transcription if isinstance(transcription, str) else transcription.text
    text = (text or "").strip()
    if not text:
        raise RuntimeError("Transcription returned no text.")
    return text


def build_retriever(text: str, embeddings: OpenAIEmbeddings):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RETRIEVER_CHUNK_CHARS,
        chunk_overlap=RETRIEVER_CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(text)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})


def format_docs(docs) -> str:
    return "\n\n".join(
        f"[Excerpt {index}]\n{doc.page_content}"
        for index, doc in enumerate(docs, start=1)
    )


def summarize_transcript(
    transcript: str,
    llm,
    max_stuff_chars: int = MAX_STUFF_CHARS,
    map_chunk_chars: int = MAP_CHUNK_CHARS,
    progress_callback: Callable[[str], None] | None = None,
) -> str:
    def notify(message: str) -> None:
        print(message)
        if progress_callback is not None:
            progress_callback(message)

    if len(transcript) <= max_stuff_chars:
        notify("Summarizing transcript...")
        chain = SUMMARY_PROMPT | llm | StrOutputParser()
        summary = chain.invoke({"transcript": transcript}).strip()
        if not summary:
            raise RuntimeError("The model returned an empty summary.")
        return summary

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=map_chunk_chars,
        chunk_overlap=MAP_CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(transcript)
    map_chain = MAP_PROMPT | llm | StrOutputParser()
    section_summaries = []

    for index, chunk in enumerate(chunks, start=1):
        notify(f"Summarizing section {index}/{len(chunks)}...")
        section_summary = map_chain.invoke({"transcript": chunk}).strip()
        if section_summary:
            section_summaries.append(section_summary)

    if not section_summaries:
        raise RuntimeError("The model returned no section summaries.")

    notify("Combining section summaries...")
    reduce_chain = REDUCE_PROMPT | llm | StrOutputParser()
    summary = reduce_chain.invoke(
        {"summaries": "\n\n".join(section_summaries)}
    ).strip()
    if not summary:
        raise RuntimeError("The model returned an empty summary.")
    return summary


def answer_question_with_docs(question: str, retriever, llm):
    docs = retriever.invoke(question)
    if not docs:
        return "I could not find that in the video.", []

    chain = QA_PROMPT | llm | StrOutputParser()
    answer = chain.invoke(
        {"context": format_docs(docs), "question": question}
    ).strip()
    return answer or "I could not find that in the video.", docs


def answer_question(question: str, retriever, llm) -> str:
    answer, _ = answer_question_with_docs(question, retriever, llm)
    return answer


def build_llm(api_key: str):
    primary = ChatOpenAI(
        model=CHAT_MODEL,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.2,
        max_tokens=2048,
        request_timeout=180,
        max_retries=3,
        extra_body={"reasoning": {"enabled": False}},
    )
    fallback = ChatOpenAI(
        model=FALLBACK_CHAT_MODEL,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.2,
        max_tokens=2048,
        request_timeout=180,
        max_retries=2,
        extra_body={"reasoning": {"enabled": False}},
    )
    return primary.with_fallbacks([fallback])


def build_embeddings(api_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        check_embedding_ctx_length=False,
        request_timeout=60,
        max_retries=3,
    )


def main() -> None:
    load_dotenv(os.path.join(BASE_DIR, ".env"))

    groq_api_key = os.getenv("GROQ_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not groq_api_key or not openrouter_api_key:
        raise SystemExit(
            "Missing GROQ_API_KEY or OPENROUTER_API_KEY. Add them to the .env file."
        )

    groq_client = Groq(
        api_key=groq_api_key,
        timeout=300.0,
        max_retries=3,
    )
    llm = build_llm(openrouter_api_key)
    embeddings = build_embeddings(openrouter_api_key)

    url = input("Tell me the video URL: ").strip()
    if not url:
        raise SystemExit("No URL provided.")

    with tempfile.TemporaryDirectory(prefix="video_agent_") as workdir:
        print("Downloading audio...")
        audio_file = download_audio(url, workdir)
        print(f"Downloaded: {os.path.basename(audio_file)}")

        print("Transcribing audio...")
        transcript = transcribe_audio(audio_file, groq_client)

    print(f"Transcript length: {len(transcript):,} characters")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(summarize_transcript(transcript, llm))

    print("\n" + "=" * 60)
    print("Building question-answering index...")
    print("=" * 60)
    try:
        retriever = build_retriever(transcript, embeddings)
    except Exception as exc:
        print(f"Question answering is unavailable: {exc}")
        return

    print("\n" + "=" * 60)
    print("ASK QUESTIONS (type 'exit' to quit)")
    print("=" * 60)

    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        try:
            print("\n" + answer_question(question, retriever, llm))
        except Exception as exc:
            print(f"\nCould not answer that question: {exc}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc
