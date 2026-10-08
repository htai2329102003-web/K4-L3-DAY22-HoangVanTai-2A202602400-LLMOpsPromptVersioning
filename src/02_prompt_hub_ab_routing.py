"""
Bước 2 — Prompt Hub & A/B Routing
===================================
NHIỆM VỤ:
  1. Viết 2 system prompt khác nhau (V1: ngắn gọn, V2: có cấu trúc)
  2. Push cả 2 lên LangSmith Prompt Hub qua client.push_prompt()
  3. Pull lại từ Hub qua client.pull_prompt()
  4. Implement A/B routing tất định: hash(request_id) % 2 → V1 hoặc V2
  5. Chạy 50 câu hỏi qua router → ≥ 50 LangSmith traces nữa

DELIVERABLE: 2 prompt version hiển thị trong Prompt Hub trên https://smith.langchain.com
"""
import sys
import hashlib
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS


# ── 1. Tên Prompt trên Hub ─────────────────────────────────────────────────
# TODO: Đổi thành tên của bạn — phải là duy nhất trong Hub của bạn
PROMPT_V1_NAME = "hoang-van-tai-2a202602400-rag-prompt-v1"
PROMPT_V2_NAME = "hoang-van-tai-2a202602400-rag-prompt-v2"


# ── 2. Định nghĩa 2 Prompt Templates ──────────────────────────────────────
# TODO: Viết SYSTEM_V1 — phong cách ngắn gọn, trả lời 2-4 câu
# Gợi ý: "Bạn là trợ lý AI hữu ích. Chỉ dùng context sau để trả lời.
#          Giữ câu trả lời ngắn gọn (2-4 câu). ..."
SYSTEM_V1 = (
    "Bạn là trợ lý AI thân thiện. Trả lời ngắn gọn trong 2-4 câu và chỉ dựa "
    "trên context được cung cấp. Nếu context không đủ thông tin, hãy nói rõ "
    "rằng bạn không biết.\n\nContext:\n{context}"
)

PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

# TODO: Viết SYSTEM_V2 — phong cách có cấu trúc, expert tone, 3-5 câu
# Gợi ý: "Bạn là chuyên gia AI. Đọc kỹ context, xác định facts liên quan,
#          viết câu trả lời rõ ràng và có tổ chức (3-5 câu). ..."
SYSTEM_V2 = (
    "Bạn là chuyên gia phân tích thông tin. Hãy đọc kỹ context, xác định các "
    "facts liên quan, rồi trình bày câu trả lời rõ ràng, có tổ chức trong 3-5 "
    "câu. Không suy đoán hoặc bổ sung thông tin ngoài context.\n\n"
    "Context:\n{context}"
)

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])


# ── 3. Push Prompts lên Prompt Hub ─────────────────────────────────────────
def push_prompts_to_hub(client: Client):
    """
    Upload cả 2 prompt templates lên LangSmith Prompt Hub.
    Gợi ý: client.push_prompt(name, object=template, description="...")
    """
    # TODO: Push PROMPT_V1 — bọc trong try/except để xử lý lỗi
    try:
        url = client.push_prompt(
            PROMPT_V1_NAME,
            object=PROMPT_V1,
            description="V1 – ngắn gọn, thân thiện, 2-4 câu",
        )
        print(f"✅ Đã push V1 → {url}")
    except Exception as e:
        if "409" in str(e) or "Nothing to commit" in str(e):
            print(f"ℹ️  V1 đã tồn tại và không thay đổi: {PROMPT_V1_NAME}")
        else:
            raise

    # TODO: Push PROMPT_V2 — bọc trong try/except
    try:
        url = client.push_prompt(
            PROMPT_V2_NAME,
            object=PROMPT_V2,
            description="V2 – chuyên gia, có cấu trúc, 3-5 câu",
        )
        print(f"✅ Đã push V2 → {url}")
    except Exception as e:
        if "409" in str(e) or "Nothing to commit" in str(e):
            print(f"ℹ️  V2 đã tồn tại và không thay đổi: {PROMPT_V2_NAME}")
        else:
            raise


# ── 4. Pull Prompts từ Prompt Hub ──────────────────────────────────────────
def pull_prompts_from_hub(client: Client) -> dict:
    """
    Tải 2 prompt từ LangSmith Prompt Hub.
    Nếu Hub không khả dụng, hàm phát sinh lỗi thay vì dùng local fallback.

    Gợi ý: client.pull_prompt(name) → ChatPromptTemplate

    Trả về: {name: ChatPromptTemplate}
    """
    prompts = {}

    # TODO: Pull PROMPT_V1_NAME từ Hub; không dùng local fallback
    prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
    print(f"↓ Đã pull '{PROMPT_V1_NAME}' từ Hub")

    # TODO: Pull PROMPT_V2_NAME từ Hub; không dùng local fallback
    prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
    print(f"↓ Đã pull '{PROMPT_V2_NAME}' từ Hub")

    return prompts


# ── 5. A/B Routing tất định ────────────────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Xác định prompt version dựa trên MD5 hash của request_id.

    Quy tắc: hash chẵn → PROMPT_V1_NAME | hash lẻ → PROMPT_V2_NAME
    TÍNH CHẤT: cùng request_id LUÔN cho cùng kết quả (deterministic).

    Gợi ý:
        hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME
    """
    # TODO: Tính MD5 hash của request_id và chuyển thành số nguyên
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)

    # TODO: Trả về PROMPT_V1_NAME nếu chẵn, PROMPT_V2_NAME nếu lẻ
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


# ── 6. Traced A/B Query ────────────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """
    Chạy RAG chain với prompt version được chọn bởi router.

    Bước:
      a) Retrieve top-3 docs từ retriever
      b) Ghép page_content thành context string
      c) Chạy (prompt | llm | StrOutputParser()).invoke({"context": ..., "question": ...})
      d) Trả về {"question": ..., "answer": ..., "version": ...}
    """
    # TODO: Retrieve docs từ retriever
    docs = retriever.invoke(question)

    # TODO: Ghép page_content thành 1 string (dùng "\n\n".join)
    context = "\n\n".join(doc.page_content for doc in docs)

    # TODO: Chạy chain và lấy answer
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "question": question,
    })

    # TODO: Trả về dict kết quả
    return {"question": question, "answer": answer, "version": version}


REQUEST_DELAY_SECONDS = 5.0
MAX_RATE_LIMIT_RETRIES = 3


def is_rate_limit_error(error: Exception) -> bool:
    """Nhận diện HTTP 429 từ các phiên bản SDK Gemini khác nhau."""
    status_code = getattr(error, "status_code", None)
    code = getattr(error, "code", None)
    return status_code == 429 or code == 429 or "429" in str(error)


# ── 7. Setup Vectorstore (tái sử dụng logic Bước 1) ───────────────────────
def setup_vectorstore():
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 8. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 2: Prompt Hub & A/B Routing")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    # TODO: Tạo LangSmith Client với API key từ config
    # Gợi ý: client = Client(api_key=config.LANGSMITH_API_KEY)
    client = Client(api_key=config.LANGSMITH_API_KEY)

    # TODO: Push cả 2 prompts lên Hub
    push_prompts_to_hub(client)

    # TODO: Pull cả 2 prompts từ Hub (dùng dict trả về)
    prompts = pull_prompts_from_hub(client)

    # Tạo vectorstore, retriever và LLM
    vectorstore = setup_vectorstore()
    # TODO: Tạo retriever từ vectorstore (k=3)
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm         = get_llm()

    # Chạy A/B routing cho tất cả câu hỏi
    v1_count, v2_count = 0, 0
    retry_count = 0
    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id  = f"req-{i:04d}"

        # TODO: Lấy version key từ request_id qua get_prompt_version()
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt      = prompts[version_key]

        # TODO: Gọi ask_ab() với đúng arguments
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            try:
                result = ask_ab(retriever, llm, prompt, question, version_tag)
                break
            except Exception as error:
                if not is_rate_limit_error(error) or attempt == MAX_RATE_LIMIT_RETRIES:
                    raise
                retry_count += 1
                backoff_seconds = REQUEST_DELAY_SECONDS * (2 ** attempt)
                print(
                    f"⚠️ Gemini rate limit ở câu {i + 1}; "
                    f"retry {attempt + 1}/{MAX_RATE_LIMIT_RETRIES} "
                    f"sau {backoff_seconds:.1f}s"
                )
                time.sleep(backoff_seconds)

        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1
        print(f"[{i+1:02d}] [prompt-{version_tag}] {question[:55]}...")
        if i + 1 < len(SAMPLE_QUESTIONS):
            time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\n📊 Routing: V1={v1_count} câu | V2={v2_count} câu | Tổng={len(SAMPLE_QUESTIONS)}")
    print(f"   Rate-limit retries: {retry_count}")
    print("✅ Bước 2 hoàn thành! Kiểm tra Prompt Hub và traces trên LangSmith.")


if __name__ == "__main__":
    main()
