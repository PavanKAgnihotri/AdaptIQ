import os, re
from groq import Groq
from qdrant_client.models import Filter, FieldCondition, MatchValue
from retrieval.ingester import qdrant, embedder

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
COLLECTION_NAME = "adaptiq_docs"
VECTOR_SIZE     = 384

# ── Prompt with anti-hallucination citation rule ──────
EXPLAIN_PROMPT = """You are an expert AI/ML interview coach.
A student answered an interview question INCORRECTLY.
Explain the correct concept clearly using the provided context.

RULES:
1. Answer using ONLY the provided context chunks
2. You MUST include at least one citation in your response
3. Citation format: [Source: <name>]
   Use the EXACT source name from the context headers
   Example: if context shows "[Source: LangChain Docs — Agents]"
   write exactly: [Source: LangChain Docs — Agents]
4. Keep under 150 words
5. End with one specific study tip
6. Be positive — explain the concept clearly
"""


class ExplanationEngine:

    def retrieve(self, query: str, category: str,
                 top_k: int = 5) -> list[dict]:
        """
        Retrieve web doc chunks — post-filter approach
        Excludes AdaptIQ Question Bank from Qdrant results
        """
        query_emb = embedder.encode(
            query, normalize_embeddings=True
        ).tolist()

        retrieve_k = top_k * 4  # fetch more, filter down

        # Try category-filtered first
        try:
            results = qdrant.query_points(
                collection_name = COLLECTION_NAME,
                query           = query_emb,
                query_filter    = Filter(must=[
                    FieldCondition(
                        key   = "category",
                        match = MatchValue(value=category)
                    )
                ]),
                limit        = retrieve_k,
                with_payload = True
            ).points
        except Exception:
            results = []

        # Fallback: no category filter
        if len(results) < 3:
            results = qdrant.query_points(
                collection_name = COLLECTION_NAME,
                query           = query_emb,
                limit           = retrieve_k,
                with_payload    = True
            ).points

        # Post-filter — exclude AdaptIQ Question Bank
        web_chunks = [
            r for r in results
            if r.payload.get("source") != "AdaptIQ Question Bank"
        ]
        static_chunks = [
            r for r in results
            if r.payload.get("source") == "AdaptIQ Question Bank"
        ]

        print(f"  📚 Web chunks found:    {len(web_chunks)}")
        print(f"  📝 Static chunks found: {len(static_chunks)}")

        # Broaden if too few web chunks
        if len(web_chunks) < 2:
            print(f"  🔍 Broadening search...")
            broad = qdrant.query_points(
                collection_name = COLLECTION_NAME,
                query           = query_emb,
                limit           = retrieve_k,
                with_payload    = True
            ).points
            web_chunks = [
                r for r in broad
                if r.payload.get("source") != "AdaptIQ Question Bank"
            ]
            print(f"  📚 After broadening: {len(web_chunks)} web chunks")

        return [
            {
                "text":     r.payload["text"],
                "source":   r.payload["source"],
                "category": r.payload.get("category", ""),
                "score":    r.score
            }
            for r in web_chunks[:top_k]
        ]

    def generate_explanation(self,
                             question_text: str,
                             correct_answer: str,
                             student_answer: str,
                             category: str,
                             static_explanation: str = "") -> dict:
        """
        Generate cited explanation — web sources first
        Anti-hallucination: prompt forces exact source names
        """

        query  = f"{category}: {question_text} correct answer"
        chunks = self.retrieve(query, category, top_k=5)

        # Static appended LAST as fallback only
        if static_explanation:
            chunks.append({
                "text":     static_explanation,
                "source":   "AdaptIQ Question Bank",
                "score":    0.5
            })

        if not chunks:
            return {
                "explanation": (
                    f"Correct answer: {correct_answer}. "
                    f"{static_explanation} "
                    f"[Source: AdaptIQ Question Bank]"
                ),
                "sources":     ["AdaptIQ Question Bank"],
                "chunks_used": 0,
                "fallback":    True
            }

        # ── Build context with EXPLICIT source headers ─────
        # This is what forces correct citation names
        context_parts = []
        for i, chunk in enumerate(chunks[:5]):
            context_parts.append(
                f"[Source: {chunk['source']}]\n"
                f"{chunk['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # ── Available sources list — stops hallucination ───
        available_sources = list(set(
            c["source"] for c in chunks[:5]
        ))
        sources_list = "\n".join(
            f"  - {s}" for s in available_sources
        )

        try:
            response = groq_client.chat.completions.create(
                model    = "llama-3.1-8b-instant",
                messages = [
                    {"role": "system",
                     "content": EXPLAIN_PROMPT},
                    {"role": "user",
                      "content": (
                          f"Question: {question_text}\n"
                          f"Correct answer: {correct_answer}\n"
                          f"Student chose: {student_answer}\n"
                          f"Category: {category}\n\n"
                          f"Context (cite these sources EXACTLY as written):\n"
                          f"{context}\n\n"
                          f"Remember: You MUST cite at least one source using "
                          f"[Source: <exact name>] format. "
                          f"Available source names: {', '.join(available_sources)}"
                      )}
                ],
                temperature = 0.1,
                max_tokens  = 300
            )
            explanation = response.choices[0].message.content

        except Exception as e:
            print(f"LLM error: {e}")
            explanation = (
                f"Correct answer: {correct_answer}. "
                f"{static_explanation} "
                f"[Source: AdaptIQ Question Bank]"
            )

        # Extract cited sources
        cited = list(set(
            re.findall(r"\[Source[^\]]*\]", explanation)
        ))

        # Validate — flag any hallucinated sources
        hallucinated = [
            s for s in cited
            if not any(
                src in s for src in available_sources
            )
        ]
        if hallucinated:
            print(f"  ⚠️  Hallucinated sources detected: {hallucinated}")

        return {
            "explanation":  explanation,
            "sources":      cited,
            "chunks_used":  len(chunks),
            "fallback":     False,
            "hallucinated": hallucinated
        }
