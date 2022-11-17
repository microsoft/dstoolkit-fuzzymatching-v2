# People Matching

## Concepts

### Skill Ontologies

Generally in people matching scenarios, people are matched job, project, or learning opportunities by matching a set of skills possessed by the person to the set of skills from the job posting, project request, or learning description. These skills can come from structured (i.e. skills database) or unstructured sources (i.e. resumes) and come from an a known set of skills (an ontology), or can be derived via a data-driven ontology. These approaches can generally be broken down into the following:

- **Traditional Knowledge Management:** Manually maintained skill ontologies (i.e. an employee skills database), contains only structured skills ratings, generally from users rating their own skills. This approach is often the most cumbersome to maintain, becomes quickly outdated, and is often subject to bias from user ratings.
- **Application Tracking Systems (ATS):** Commonly used tool for recruiting, extracts skills from unstructured candidate resumes and matches against skills extracted from job descriptions. Generally uses a skill ontology maintained by the recruiting/staffing company.
- **Personalized Expertise Finders:** The most advanced systems which derive their ontology from the structured and unstructured data sources themselves using fuzzy matching and various NLP approaches. These systems are able to bring together multiple sources to get a fuller picture of the candidate and provide more personalized results.

![Traditional Knowledge Mining vs. Expertise Finder Systems](/docs/fuzzy_matching_people_matching_skills_ontologies.png)

This diagram explains the general benefits of expertise finder systems compared to traditional knowledge management systems.

### Skill Ontology Sources

If you are building a People Matching system that requires an existing skill ontology there are a few existing sources that can be leveraged:

- **Customer's skill ontology:** if the customer is a recruiting or staffing agency, they will likely have an existing skill ontology. The benefit of using their existing ontology is that it will likely be more catered to the types of skills they recruit but the disadvantage is that it may not be as encompassing as other skill ontologies.
- **Azure Cognitive Services Named Entity Recognition (NER) Skill Entity:** [Azure Cognitive Services NER](https://docs.microsoft.com/en-us/azure/cognitive-services/text-analytics/named-entity-types?tabs=general) can extract specific entities from structured and unstructured data sources using a pre-built NER model. In version 3.0 of NER, a skills entity was added, the result of work done by the Worldwide Learning Innovation Team. This taxonomy was built using a record linkage model built on open sources including Coursera, Microsoft Academic Graph, Github Featured Topics, Stackshare Tools, and more. Examples of a job matching demo and product built by the WWL Innovation team using this taxonomy can be found here [Using Azure Search custom skills to create personalized job recommendations](https://azure.microsoft.com/en-us/blog/using-azure-search-custom-skills-to-create-personalized-job-recommendations/) and Project Lagro (find introduction deck in the *docs* folder).
- **LinkedIn Enterprise Standardized Skills Ontology** [LinkedIn](https://developer.linkedin.com/docs/ref/v2/standardized-data/skills) maintains a set of ~35K standardized skills based on skills used by LinkedIn's 350 million member base. These skills are used for LinkedIn's skill recommendations, job matching, and recruiter tools. Access to [LinkedIn's Standardized Data Skill API](https://developer.linkedin.com/docs/ref/v2/standardized-data/skills) requires an enterprise license and strict legal contract between your customer and LinkedIn as this is regarded as LinkedIn IP and can only be used under limited terms.

### Derived Ontology Sources

For building People Matching scenarios using derived skill ontologies, it is important to first define what skill expertise means and then to identify possible data sources for the different components of skill expertise. Based on research on the design of expertise finding systems, skill expertise can be generally divided into the 4 areas shown below:

![Derived Ontology Sources](/docs/fuzzy_matching_people_matching_ontology_sources.png)

In addition, a list of some of the common data sources used (both structured and unstructured) that represent explicit or implicit indicators of skill expertise can be found below:

![Implicit Indicators vs. Explicit Indicators](/docs/fuzzy_matching_people_matching_indicators.png)

Additional details on approaches and a demo for automated expertise finding can be found in the *docs* folder.

---

## Architecture

### First-Party Azure Architecture

![First-Party Azure Architecture](/architecture/fuzzy_matching_people_matching_architecture.png)

A high-level people matching architecture using first party Azure tools will generally follow the pattern above:

1. Ingest structured data from sources like HR and candidate databases and line of business (LOB) applications as well as unstructured documents like resumes and job descriptions using [Azure Data Factory](https://azure.microsoft.com/en-in/services/data-factory/#features).
2. Store the structured data as tables in [Azure SQL](https://azure.microsoft.com/en-us/services/sql-database/) and the unstructured data as blobs in [Azure Storage](https://azure.microsoft.com/en-us/services/storage/).
3. Use [Azure Cognitive Search](https://docs.microsoft.com/en-us/azure/search/search-what-is-azure-search) and the [Named Entity Recognition (NER) Cognitive Skill](https://docs.microsoft.com/en-us/azure/search/cognitive-search-skill-entity-recognition) to index the structured and unstructured data and extract skills using the [skill entity](https://docs.microsoft.com/en-us/azure/cognitive-services/text-analytics/named-entity-types?tabs=general). By enabling [fuzzy search](https://docs.microsoft.com/en-us/azure/search/search-query-fuzzy) in Azure Cognitive Search, the search service will perform fuzzy matching using [Damerau Levenshtein Distance](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance). Best practices for implementing Azure Cognitive Search can be found in the [Knowledge Mining repo](https://github.com/microsoft/dstoolkit-km-solution-accelerator).
4. Integrate Azure Cognitive Search into an application through the Search APIs or into an intelligent bot via [Azure Bot Service](https://azure.microsoft.com/en-us/services/bot-service/). An example of using a Bot interface for Azure Cognitive Search can be found here: [Learn AI Knowledge Mining Bootcamp](https://azure.github.io/LearnAI-KnowledgeMiningBootcamp/). Natural language queries sent to the Cognitive Search APIs via the application or bot will be indexed and matched against the skills extracted from the provided data sources to find the best match. Alternatively, users can upload documents such as a resume in a user interface which will be passed through Cognitive Search to extract skills, index the text, and then match against the skills from job description documents. This blog post and demo from the WWL Innovation team demonstrates the latter approach here: [Using Azure Search custom skills to create personalized job recommendations](https://azure.microsoft.com/en-us/blog/using-azure-search-custom-skills-to-create-personalized-job-recommendations/).

### Custom Open Source Software (OSS) Architecture

![First-Party Azure Architecture](/architecture/fuzzy_matching_people_matching_oss_architecture.png)

If a more custom approach is needed for skill extraction and/or incorporating custom ranking/personalization features, an architecture that supports additional open source logic may be required. While the overall architectural components that are recommended do not change, a custom skill in Azure Cognitive Search can be used to handle custom skill extraction and a custom scoring profile can be created to handle customized ranking. The process for this pattern will look like this:

1. Ingest structured data from sources like HR and candidate databases and line of business (LOB) applications as well as unstructured documents like resumes and job descriptions using [Azure Data Factory](https://azure.microsoft.com/en-in/services/data-factory/#features).
2. Store the structured data as tables in [Azure SQL](https://azure.microsoft.com/en-us/services/sql-database/) and the unstructured data as blobs in [Azure Storage](https://azure.microsoft.com/en-us/services/storage/).
3. Use [Azure Cognitive Search](https://docs.microsoft.com/en-us/azure/search/search-what-is-azure-search) and a custom skill will allow you to leverage the compute from the Azure Function to incorporate custom skill extraction (see next section) for scenarios where you plan to use ontologies outside of the Cognitive Services NER skill entity or where a derived ontology is built. [Custom scoring profiles](https://docs.microsoft.com/en-us/azure/search/index-add-scoring-profiles) can be used along with data from personalization features such as geo location, interests, and user preferences to provide custom ranking and more personalized search results (see next section).
4. Integrate Azure Cognitive Search into an application through the Search APIs or into an intelligent bot via [Azure Bot Service](https://azure.microsoft.com/en-us/services/bot-service/). An example of using a Bot interface for Azure Cognitive Search can be found here: [Learn AI Knowledge Mining Bootcamp](https://azure.github.io/LearnAI-KnowledgeMiningBootcamp/). Natural language queries sent to the Cognitive Search APIs via the application or bot will be indexed and matched against the skills extracted from the provided data sources to find the best match. Alternatively, users can upload documents such as a resume in a user interface which will be passed through Cognitive Search to extract skills, index the text, and then match against the skills from job description documents. This blog post and demo from the WWL Innovation team demonstrates the latter approach here: [Using Azure Search custom skills to create personalized job recommendations](https://azure.microsoft.com/en-us/blog/using-azure-search-custom-skills-to-create-personalized-job-recommendations/).

---

## Techniques

### Custom Skill Extraction

For custom skill extraction, there are two approaches: an unsupervised approach, and a supervised machine learning approach that requires a labelled training dataset of skills. The unsupervised approach requires an existing skill ontology while the supervised machine learning approach can can be used to generate a data-driven ontology.

### Unsupervised Skill Extraction

In unsupervised custom skill extraction an existing skill ontology such as the [LinkedIn Enterprise Standardized Skills Ontology](https://developer.linkedin.com/docs/ref/v2/standardized-data/skills) or the customer's own skill ontology are required. This process involves the following steps:

1. Pre-process the text data from the candidate data sources (resume, HR database, etc) and job data source (job description, etc.) using techniques identified in the [Data Management](/docs/DataManagement.md) document, section *pre-processing techniques* to do things like removing stopwords, trimming words to their root, and converting to n-gram tokens.
2. (Optional) Leverage word embeddings such as [BERT](https://en.wikipedia.org/wiki/BERT_(language_model)), [ELMo](https://allennlp.org/elmo), [Word2Vec](https://en.wikipedia.org/wiki/Word2vec) to convert both the skills in the ontology and the words or n-grams in the text data into vectors that represent their semantic similarity to better capture words with similar meanings but different spellings.
3. Leverage fuzzy matching techniques (see: [Fuzzy Matching Techniques](/docs/DataManagement.md) (section *Techniques*) to perform fuzzy matching of each word or vector from the candidate date sources and job data sources against each word or vector representation of the skill ontology. Blocking and other techniques will be needed to reduce the computational intensity of this task.
4. Perform this skill extraction as part of a [python Azure Function custom skill](https://docs.microsoft.com/en-us/azure/search/cognitive-search-custom-skill-python) in Azure Cognitive Search. Return the approximate skill matches to the search index to be used by the search service or [custom scoring profile](https://docs.microsoft.com/en-us/azure/search/index-add-scoring-profiles).

### Supervised Skill Extraction

In supervised custom skill extraction, a custom Named Entity Recognition (NER) model and fuzzy matching is used to generate a data-driven skill ontology. This process involves the following steps:

1. Generate an annotated training set of documents and and its associated skills to train a NER model. LUIS for Documents (LUIS-D) is an upcoming Azure service that helps with annotating and training a custom NER model. You can sign up for the private preview [here](https://forms.office.com/Pages/ResponsePage.aspx?id=v4j5cvGGr0GRqy180BHbR2PqAWXp87hBnu6-EavmD4lUNTEzRkJIMEI1S0c1VTlGSDRMT0k0RVFEMC4u).
2. Train a custom NER model using the annotated training set using a python library such as [SpaCy](https://spacy.io/usage/linguistic-features#named-entities) or using a word embedding such as [BERT](https://en.wikipedia.org/wiki/BERT_(language_model)), [ELMo](https://allennlp.org/elmo), [Word2Vec](https://en.wikipedia.org/wiki/Word2vec) for a more context-aware NER model. Use this model to extract skill entities from both the candidate and job data sources.
3. Leverage fuzzy matching techniques (see: [Fuzzy Matching Techniques](/docs/DataManagement.md) (section *Techniques*) to perform fuzzy matching of each word or vector from the candidate date sources and job data sources against each other to conflate potential skills down to a skill ontology that includes synonyms that can be included in the Azure Cognitive Search index.
4. Incorporate the NER model and fuzzy matching as part of a [python Azure Function custom skill](https://docs.microsoft.com/en-us/azure/search/cognitive-search-custom-skill-python) in Azure Cognitive Search. Return the approximate skill matches to the search index to be used by the search service or [custom scoring profile](https://docs.microsoft.com/en-us/azure/search/index-add-scoring-profiles).

### Personalized Ranking

[Custom scoring profile](https://docs.microsoft.com/en-us/azure/search/index-add-scoring-profiles) in Azure Cognitive Search can be used to incorporate custom logic for ranking and personalizing search results. By adding personalization features such as geographic closeness, interests, and user preferences datasets and incorporating matches to those criteria from the job postings (or projects or learning paths), the scoring profiles can be used to weigh these factors in addition to purely skill matches and provide a more catered experience to the end user.

More sophisticated custom ranking models such as [Learn-to-Rank Functions](https://en.wikipedia.org/wiki/Learning_to_rank) like PageRank and WordNet can also be leveraged as a custom skill and the output can be incorporated into the scoring profile but these often require large volumes of training data. Reinforcement learning models can also be incorporated as custom skills to leverage clickstream data and user feedback to improve search rankings.

## UI/UX Considerations

In People Matching solutions, the user experience (UX) and user interface (UI) represent one of the largest components of the overall solution and should not be designed in isolation from the underlying data solution. Some important user experience considerations include the following:

- **Facets and filters** for search based solutions to allow users to filter and browse search results. These can be built off of the data in the search index and configured in [Azure Cognitive Search](https://docs.microsoft.com/en-us/azure/search/search-filters).
- **Transparency in the UX** is important for end users to understand why they are seeing certain results or certain errors. An example can be shown below where the the header above the document provides context on why it is being recommended. Microsoft's [guidelines for human-AI interaction](https://docs.microsoft.com/en-us/ai/guidelines-human-ai-interaction/) provide many useful examples and best practices for good use of human-AI interaction in applications.
![Transparency in the UX](/docs/fuzzy_matching_people_matching_transparency.png)
- There are important **considerations on asking the user for information vs. inferring information**. While a more streamlined UX with minimal prompts and profile requirements for users will reduce the barriers to using the application, there will be a tradeoff in potential limitations to the types of intelligence that can be provided. When possible, associating a user with different source of truth datasets like HR databases or even external profiles like LinkedIn and Facebook to pre-fill information can help to ease this burden and prevent users from being required to manually enter too much data on their end.

## AI Ethics
It is always important to consider ethics when designing AI systems, particularly in people matching scenarios where there are high stakes consequences to the end user of the system. For more resources on AI ethical considerations visit the [AETHER Page](https://microsoft.sharepoint.com/teams/Aether/SitePages/Resources.aspx).

