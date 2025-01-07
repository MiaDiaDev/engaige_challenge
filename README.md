Bitte die requirements.txt installieren und valide Keys in der .env hinterlegen.


Verbesserungsmöglichkeiten und Ausblick:

Extractive QA: Wörtliche Wiedergabe (Zitieren) von Textpassagen könnte besser zu Gesetzestexten passen. Falls Infos an mehreren Stellen in den Dokumenten stehen, scheint trotzdem ein Zusammenfassungsschritt unter Verwendung eines LLMs nötig. Hier besteht somit wieder die Gefahr von Missinterpretation, Missachtung/Fehlen von Kontext oder Halluzinationen.

Hierarchisches Splitting: Baumstruktur nutzen um Zuordnung von Abschnitten zu Paragraphen verdeutlichen (Wichtigkeit abhängig von Art der Anfrage) -> bessere Metadaten https://docs.haystack.deepset.ai/reference/experimental-splitters-api 
oder Graph RAG z.B. https://github.com/TilmanLudewigtHaufe/GraphAugmented-Legal-RAG  

zusätzlich NER identifizieren: durch spaCy oder LLM zur Verwendung als Metadaten erzeugen und bei Inference danach Filtern; weitere Metadaten Filter

performanetere/spezifischere Embeddings z.B. ModernBert oder auf Gesetzestexte spezialisiertes Modell 

Agentic RAG: nachfragen, wenn Anfrage nicht präzise genug oder bei vers. Gesetzesteilbereichen; Query Routing nach intent (bei Erweiterung der Anwendung); Evaluationsloop, ob Anfrage ausreichend beantwortet wurde und ggfs. mehr Informationen anfordern

weitere Evaluationsmetriken & Benchmarking (z.B. https://arxiv.org/html/2408.10343)