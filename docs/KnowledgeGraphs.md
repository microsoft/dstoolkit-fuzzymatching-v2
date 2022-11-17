# Knowledge Graphs

Also referred to as: semantic knowledge graphs, enterprise knowledge graphs, semantic data lake

## Opportunity

Knowledge graphs allow you organize unstructured and structured data in a contextually aware, domain-specific graph.

## Value Statement

Uncover derived insights from large volumes of domain-specific structured and unstructured data sources to be able to answer more complex, contextually-aware questions from an organization's data. Examples of insights include:

- contextually-aware recommendations
- improved organization search, connecting employees to associated documents, M365 data, organizational hierarchy, and more ([Microsoft Search in Bing](https://www.bing.com/business/explore))
- contextually aware and more personalized people matching, and intelligent bot scenarios that take advantage of derived insights
- better alignment of structured and unstructured data sources
- increased transparency and visualization capabilities into the connections between organizational data

---

## Architecture

![Architecture](/architecture/fuzzy_matching_knowledge_graphs_architecture.png)

This architecture was built by Salim Naim (Architect Manager) and Ahmar Kazi (Solution Architect) for a knowledge graph solution on Azure using Cosmos DB and the Gremlin API for JPMorgan. You can learn more about this architecture and example code here: [Deep Fin: Knowledge Graphs](https://github.com/salimngit/DeepFin-Series-JPMorgan/blob/master/Knowledge%20Graph/Knowledge%20Graph.ipynb).

![Architecture](/architecture/fuzzy_matching_knowledge_graphs_alt_architecture.png)

An alternative architecture comes from the Trusted and Connected Data Services Team in CSEO. In this [MLADS talk](https://msit.microsoftstream.com/video/2425a1ff-0400-a936-fd76-f1eaa68afd5c?channelId=e34ba1ff-0400-a936-7258-f1eaa6716ffe) and deck (*MLADSDataUnificationAtScale.pptx* found in the *docs* folder) Danny Godbout (Data Scientist) discuss an approach for fuzzy matching that uses graph frames in Databricks Spark to visualize and derive insights from a record linkage problem. This architecture pattern can form a component of a larger of a knowledge graph solution by handling the fuzzy matching of entities from various data source at scale and representing the results of that fuzzy matching into a Spark graph frame that can form the foundation of an enterprise knowledge graph.

### Note 
While Knowledge Graph solutions can be incredibly valuable, it is typically a complex architecture and involves numerous challenges to address including scalability considerations, ingestion of large volumes of structured and unstructured data sources, fuzzy matching and pre-processing these data sources, and building the graph ontology model.

---

## Azure SaaS Solutions

Microsoft has two SaaS solutions on Azure for leveraging structured and unstructured data from an organizations M365 data (SharePoint, Office, Teams, Outlook) to generate valuable insights and improve productivity which leverages knowledge graph technology under the hood: [Microsoft Search in Bing](https://www.bing.com/business/explore) and [Project Cortex](https://resources.techcommunity.microsoft.com/project-cortex-microsoft-365/).

### Microsoft Search in Bing

[Microsoft Search in Bing](https://www.bing.com/business/explore) allows you to search for people, groups, and answers within your organization, powered by Bing Satori Knowledge Graph.

![Microsoft Search in Bing](/docs/fuzzy_matching_knowledge_graphs_search_in_bing.png)

### Project Cortex

[Project Cortex](https://resources.techcommunity.microsoft.com/project-cortex-microsoft-365/) applies AI to automatically organize your content, and delivers innovative experiences—topic cards, topic pages and knowledge centers—in Office, Outlook and Microsoft Teams.

![Microsoft Search in Bing](/docs/fuzzy_matching_knowledge_graphs_project_cortex.png)


