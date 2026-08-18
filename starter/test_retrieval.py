from src.rag import load_vectorstore


vectorstore = load_vectorstore()

print("Chunks:", vectorstore._collection.count())


queries = [
    "What is OAuth2?",
    "What is the X-Upwork-API-TenantId header?",
    "Who won the FIFA World Cup?",
]


for query in queries:

    print()
    print("=" * 70)
    print("QUERY:", query)
    print("=" * 70)

    results = vectorstore.similarity_search_with_relevance_scores(
        query,
        k=5,
    )

    for i, (doc, score) in enumerate(results, 1):

        page = doc.metadata.get(
            "page",
            "?",
        )

        text = (
            doc.page_content[:700]
            .replace("\n", " ")
        )

        print()
        print(
            f"Result {i} | "
            f"Score: {score:.3f} | "
            f"Page: {page}"
        )

        print(text)